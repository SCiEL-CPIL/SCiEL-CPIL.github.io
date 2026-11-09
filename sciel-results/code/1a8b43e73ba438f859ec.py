"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16

# EVOLVE-BLOCK-START
# Fast bitset-based zero-one principle verifier using Python ints (64-bit chunks)
# Each wire state is a 65536-bit integer; bit k is 1 iff wire outputs 1 for input k.

# Precompute initial wire masks as Python ints
# Wire i: bit k is 1 iff (k >> i) & 1 == 1
def _build_initial_bitmasks():
    masks = []
    for i in range(N):
        m = 0
        block = 1 << i
        period = 1 << (i + 1)
        for k in range(65536):
            if (k >> i) & 1:
                m |= (1 << k)
        masks.append(m)
    return masks

# Faster construction using pattern repetition
def _build_initial_bitmasks_fast():
    masks = []
    for i in range(N):
        block = 1 << i
        period = 1 << (i + 1)
        # Build one period's worth of bits
        period_mask = 0
        for k in range(block, period):
            period_mask |= (1 << k)
        # Tile across all 65536 bits
        m = 0
        offset = 0
        while offset < 65536:
            m |= (period_mask << offset)
            offset += period
        masks.append(m)
    return masks

_INITIAL_BITMASKS = _build_initial_bitmasks_fast()

def _verify_network(comps):
    """Verify sorting network using zero-one principle with Python int bitmasks."""
    if not comps:
        return True
    wires = list(_INITIAL_BITMASKS)
    for (i, j) in comps:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b      # min
        wires[j] = a | b      # max
    
    # Check sorted: wire[i] is subset of wire[i+1] (bitwise)
    for i in range(15):
        if wires[i] & ~wires[i+1]:
            return False
    return True

def _batcher_oe_merge_sort(n):
    """Build Batcher's odd-even merge sort network."""
    def oe_merge_network(n):
        """Odd-even merge network for 2n inputs (n from top half, n from bottom half)."""
        if n == 1:
            return [(0, 1)]
        # Compare (i, n+i) for i in 0..n-1
        comps = [(i, n + i) for i in range(n)]
        # Recursively merge even-indexed and odd-indexed subsequences
        even_merge = oe_merge_network(n // 2)
        odd_merge = oe_merge_network(n // 2)
        # Apply even merge on even positions
        for (a, b) in even_merge:
            comps.append((2 * a, 2 * b))
        # Apply odd merge on odd positions
        for (a, b) in odd_merge:
            comps.append((2 * a + 1, 2 * b + 1))
        return comps
    
    def sort_network(n):
        """Sorting network using odd-even merge sort."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        if n % 2 == 1:
            base = sort_network(n - 1)
            extra = [(i, n - 1) for i in range(n - 1)]
            return base + extra
        half = n // 2
        left = sort_network(half)
        right = [(i + half, j + half) for (i, j) in sort_network(half)]
        merge = oe_merge_network(half)
        return left + right + merge
    
    return sort_network(n)

def _compute_wire_influence(comps):
    """
    Compute a wire-influence metric: for each wire, how many comparators it participates in.
    Also compute a pairwise influence score: how "useful" a comparator (i,j) is,
    based on how much the two wires disagree in their current bitmasks.
    Returns a dict mapping (i,j) -> influence_score.
    """
    # Compute final wire states
    wires = list(_INITIAL_BITMASKS)
    for (i, j) in comps:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b
        wires[j] = a | b
    
    # For each comparator, compute how much it "matters"
    # A comparator (i,j) is important if before the comparator, wire i had 1s that wire j didn't
    # and wire j had 1s that wire i didn't (i.e., they disagree)
    influence = {}
    wires2 = list(_INITIAL_BITMASKS)
    for (i, j) in comps:
        a = wires2[i]
        b = wires2[j]
        # Disagreement: bits where they differ
        disagree = a ^ b
        # Count bits
        score = bin(disagree).count('1')
        influence[(i, j)] = score
        wires2[i] = a & b
        wires2[j] = a | b
    
    return influence

def _simulated_annealing(comps, deadline):
    """
    Simulated annealing to minimize comparator count.
    State: list of comparators.
    Energy: comparator count (we want to minimize, but must maintain correctness).
    Moves:
      - Remove a comparator (energy decreases by 1 if still correct)
      - Swap two comparators (energy unchanged, may enable future removals)
      - Replace a comparator with a different one (guided by wire influence)
      - Insert a comparator (energy increases by 1, but may enable future removals)
    """
    random.seed(42)
    best = list(comps)
    best_count = len(best)
    current = list(comps)
    current_count = len(current)
    
    # Temperature schedule: start high, cool exponentially
    T0 = 10.0
    T_min = 0.1
    # We'll run until deadline, with temperature cooling based on iterations
    # Estimate ~500000 iterations in 80s -> cool over that range
    
    start_time = time.time()
    iters = 0
    no_improve = 0
    max_no_improve = 5000
    
    # Precompute all valid pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Compute initial influence
    influence = _compute_wire_influence(current)
    
    while time.time() < deadline:
        T = max(T_min, T0 * (1.0 - iters / 500000.0))
        
        # Choose mutation type
        r = random.random()
        
        if r < 0.5:
            # Remove a comparator
            if len(current) <= 1:
                iters += 1
                continue
            idx = random.randint(0, len(current) - 1)
            trial = current[:idx] + current[idx+1:]
            if _verify_network(trial):
                # Accept: energy decreased
                current = trial
                current_count = len(current)
                no_improve = 0
                if current_count < best_count:
                    best = list(current)
                    best_count = current_count
            else:
                # Reject (energy increased: network broken)
                # In SA we could accept with probability exp(-delta/T), but broken networks
                # are not valid states, so we always reject
                no_improve += 1
                
        elif r < 0.75:
            # Swap two comparators
            if len(current) < 2:
                iters += 1
                continue
            i, j = random.sample(range(len(current)), 2)
            trial = list(current)
            trial[i], trial[j] = trial[j], trial[i]
            if _verify_network(trial):
                # Same energy, accept with some probability (to escape local optima)
                if random.random() < 0.3:
                    current = trial
                    no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        elif r < 0.9:
            # Replace a comparator with a different one guided by influence
            if not current:
                iters += 1
                continue
            idx = random.randint(0, len(current) - 1)
            old_comp = current[idx]
            # Pick a new comparator: prefer pairs with high influence scores
            # or pairs not currently in the network
            in_network = set(current)
            candidates = [p for p in all_pairs if p not in in_network]
            if not candidates:
                iters += 1
                continue
            # Weight by influence: higher influence = more likely to be useful
            # Use influence of the old comparator's wires as a guide
            oi, oj = old_comp
            # Score candidates by how much they overlap with old comparator's wires
            scored = []
            for p in candidates:
                pi, pj = p
                overlap = 0
                if pi == oi or pi == oj:
                    overlap += 1
                if pj == oi or pj == oj:
                    overlap += 1
                scored.append((overlap, p))
            # Pick randomly among top candidates
            scored.sort(key=lambda x: -x[0])
            top_k = min(10, len(scored))
            _, new_comp = random.choice(scored[:top_k])
            trial = list(current)
            trial[idx] = new_comp
            if _verify_network(trial):
                current = trial
                no_improve += 1
            else:
                no_improve += 1
                
        else:
            # Insert a new comparator (exploration)
            # Only accept if it leads to a state from which we can remove more
            in_network = set(current)
            candidates = [p for p in all_pairs if p not in in_network]
            if not candidates:
                iters += 1
                continue
            new_comp = random.choice(candidates)
            pos = random.randint(0, len(current))
            trial = current[:pos] + [new_comp] + current[pos:]
            if _verify_network(trial):
                # Inserted a comparator, count increased
                # Accept with probability based on temperature (to allow exploration)
                delta = 1  # energy increase
                if random.random() < min(1.0, T / max(delta, 0.1)):
                    current = trial
                    current_count = len(current)
                    no_improve += 1
                else:
                    no_improve += 1
            else:
                no_improve += 1
        
        iters += 1
        
        if no_improve > max_no_improve:
            # Restart with best solution
            current = list(best)
            current_count = best_count
            no_improve = 0
    
    return best

def _greedy_polish(comps, deadline):
    """
    Greedy removal polish: repeatedly try to remove each comparator.
    Use the fast bitset verifier.
    """
    best = list(comps)
    best_count = len(best)
    
    max_passes = 100
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        # Try removing each comparator in random order
        order = list(range(len(best)))
        random.shuffle(order)
        for idx in order:
            if time.time() > deadline:
                break
            
            trial = best[:idx] + best[idx+1:]
            if _verify_network(trial):
                best = trial
                best_count = len(best)
                improved = True
                break  # Restart pass since list changed
        
        if not improved:
            break
    
    return best

def construct():
    # Start with Batcher's odd-even merge sort
    comps = _batcher_oe_merge_sort(N)
    
    if not _verify_network(comps):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Deadline with safety margin
    deadline = time.time() + 85.0
    
    # Phase 1: Simulated annealing (main search)
    best = _simulated_annealing(comps, deadline - 10.0)
    
    # Phase 2: Greedy polish
    remaining = deadline - time.time()
    if remaining > 3.0:
        best = _greedy_polish(best, deadline)
    
    return best
# EVOLVE-BLOCK-END
