import numpy as np
import time
import random
from collections import defaultdict, deque

N = 16

# EVOLVE-BLOCK-START
def construct():
    # Batcher's odd-even mergesort network for N=16 (60 comparators)
    def batcher_odd_even_merge_sort(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        k = n // 2
        even_net = batcher_odd_even_merge_sort(k)
        odd_net = batcher_odd_even_merge_sort(k)
        even_mapped = [(2*i, 2*j) for i, j in even_net]
        odd_mapped = [(2*i+1, 2*j+1) for i, j in odd_net]
        merge_net = batcher_odd_even_merge(k)
        return even_mapped + odd_mapped + merge_net

    def batcher_odd_even_merge(k):
        if k == 1:
            return [(0, 1)]
        first = [(2*i, 2*i+1) for i in range(k)]
        k2 = k // 2
        even_merge = batcher_odd_even_merge(k2)
        odd_merge = batcher_odd_even_merge(k2)
        even_mapped = [(2*i, 2*j) for i, j in even_merge]
        odd_mapped = [(2*i+1, 2*j+1) for i, j in odd_merge]
        return first + even_mapped + odd_mapped

    batcher_net = batcher_odd_even_merge_sort(N)
    
    # Pre-compute inputs and targets
    inputs = np.arange(2**N, dtype=np.uint32)
    popcounts = np.array([bin(x).count('1') for x in range(2**N)], dtype=np.uint8)
    targets = np.array([((1 << k) - 1) << (N - k) if k > 0 else 0 for k in popcounts], dtype=np.uint32)

    def verify(net):
        masks = inputs.copy()
        for i, j in net:
            bit_i = 1 << i
            bit_j = 1 << j
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        return np.array_equal(masks, targets)

    def simulate_and_track(net):
        """Simulate network, tracking per-comparator firing masks and per-wire state history.
        Returns (valid, fired_list, wire_states) where wire_states[c][w] is the value on wire w
        just before comparator c fires."""
        n_comp = len(net)
        masks = inputs.copy()
        fired = [None] * n_comp
        wire_states = [None] * n_comp
        
        for c, (i, j) in enumerate(net):
            bit_i = 1 << i
            bit_j = 1 << j
            # Record state before this comparator
            wire_states[c] = masks.copy()
            cond = ((masks >> i) & 1) & ~((masks >> j) & 1)
            fired[c] = cond
            if np.any(cond):
                masks = np.where(cond, masks ^ bit_i ^ bit_j, masks)
        
        valid = np.array_equal(masks, targets)
        return valid, fired, wire_states

    # Verify Batcher network
    if not verify(batcher_net):
        initial_net = [(i, i + 1) for phase in range(N)
                       for i in range(phase % 2, N - 1, 2)]
        return initial_net

    current_net = list(batcher_net)
    best_net = list(batcher_net)
    best_len = len(current_net)
    
    start_time = time.time()
    deadline = 55.0
    random.seed(42)
    
    def build_dependency_graph(net):
        """Build dependency graph: edge a->b means removing a might break b.
        Comparator b depends on a if there exists an input where:
          - a fires before b in the network
          - b's firing condition at b's position depends on a having fired
        We approximate: a is a dependency of b if for some input,
        a fires and without a, b would not fire (or would fire differently).
        
        More precisely: b depends on a if there exists input x such that:
          1. a fires on x in the full network
          2. In the network without a, the state at b's position differs for x
             in a way that changes whether b fires.
        
        Simplified check: For each pair (a, b) with a before b, check if
        there's an input where a fires AND the wire states that b reads are
        affected by a's action.
        """
        n = len(net)
        valid, fired, wire_states = simulate_and_track(net)
        if not valid:
            return None, None
        
        # For each comparator b, determine which earlier comparators a
        # could potentially affect b's behavior.
        # A comparator a affects b if:
        #   - a is before b in the sequence
        #   - there exists an input x where a fires
        #   - and the values on b's wires (i_b, j_b) at b's position
        #     differ between the full network and the network without a
        #     for at least one such x.
        
        # Efficient approach: For each comparator a, compute which wires it
        # modifies. Then for each later comparator b, if b reads any wire
        # that a modifies, AND a fires for some input where b also reads
        # that wire in a state that matters, then a is a dependency of b.
        
        # Even more efficient: Precompute for each comparator a, the set of
        # (input, wire) pairs where a changes the wire value.
        # Then for comparator b reading wires (i_b, j_b), a is a dependency
        # if there exists an input x where a changes wire i_b or j_b,
        # AND in the full network b fires on x (or the state matters).
        
        # Build dependency sets
        dep_of = defaultdict(set)  # dep_of[b] = set of a's that b depends on
        
        for b in range(n):
            ib, jb = net[b]
            # For each earlier comparator a, check if it can affect b
            for a in range(b):
                ia, ja = net[a]
                # a can only affect b if a modifies one of b's wires
                if ia not in (ib, jb) and ja not in (ib, jb):
                    continue
                # Check if there's an input where a fires AND the state
                # that b sees is affected.
                # a fires on inputs where fired[a] is True.
                # We need: exists x in fired[a] such that the value on
                # ib or jb at position b differs with/without a.
                # Approximation: if a fires on any input where b also fires
                # or where the relevant wire has a specific value, it's a dep.
                # Conservative: mark as dependency if a fires at all and
                # modifies a wire b reads.
                if np.any(fired[a]):
                    dep_of[b].add(a)
        
        return dep_of, fired

    def find_removable(net, dep_of, fired):
        """Find comparators that are safe to remove using dependency graph.
        A comparator is removable if:
        1. It doesn't fire (trivial)
        2. No other comparator depends on it (it's a source in dep graph)
        3. Even if some depend on it, verify removal directly.
        """
        n = len(net)
        # Reverse: for each a, which b's depend on a?
        dependents = defaultdict(set)
        for b in range(n):
            for a in dep_of.get(b, set()):
                dependents[a].add(b)
        
        # Comparators with no dependents are candidates
        no_dependents = [a for a in range(n) if a not in dependents]
        
        # Also include non-firing comparators
        non_firing = [a for a in range(n) if not np.any(fired[a])]
        
        candidates = list(set(no_dependents) | set(non_firing))
        
        # Sort by number of dependents (fewer = safer)
        candidates.sort(key=lambda a: len(dependents.get(a, set())))
        
        return candidates

    # Main optimization loop
    while time.time() - start_time < deadline:
        net_len = len(current_net)
        if net_len <= 40:
            break
        
        # Get firing and dependency info
        valid, fired, _ = simulate_and_track(current_net)
        if not valid:
            current_net = list(best_net)
            continue
        
        # Build dependency graph
        dep_of, fired_dep = build_dependency_graph(current_net)
        if dep_of is None:
            current_net = list(best_net)
            continue
        
        # Find removable candidates
        candidates = find_removable(current_net, dep_of, fired)
        
        found_removal = False
        for idx in candidates:
            if time.time() - start_time > deadline - 5:
                break
            candidate = current_net[:idx] + current_net[idx+1:]
            if verify(candidate):
                current_net = candidate
                if len(current_net) < best_len:
                    best_net = current_net
                    best_len = len(current_net)
                found_removal = True
                break
        
        if not found_removal:
            # Try more aggressive removals: pick from all comparators
            # sorted by fire count
            fire_counts = [np.sum(fired[c]) for c in range(net_len)]
            order = sorted(range(net_len), key=lambda c: fire_counts[c])
            random.shuffle(order)
            order = sorted(order, key=lambda c: (fire_counts[c] > 500, random.random()))
            
            for idx in order[:min(8, net_len)]:
                if time.time() - start_time > deadline - 5:
                    break
                candidate = current_net[:idx] + current_net[idx+1:]
                if verify(candidate):
                    current_net = candidate
                    if len(current_net) < best_len:
                        best_net = current_net
                        best_len = len(current_net)
                    found_removal = True
                    break
            
            if not found_removal:
                # Try swaps to escape local minima
                if net_len > 2:
                    for _ in range(3):
                        i1 = random.randint(0, net_len - 1)
                        i2 = random.randint(0, net_len - 1)
                        if i1 != i2:
                            candidate = current_net.copy()
                            candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                            if verify(candidate):
                                current_net = candidate
                                break
    
    return best_net
# EVOLVE-BLOCK-END
