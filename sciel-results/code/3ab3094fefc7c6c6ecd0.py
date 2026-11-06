"""Size-efficient sorting networks on 17 wires – SAT-based structural search."""
import time
import random
import numpy as np

N = 17
MAX_COMP = 4096
TIME_LIMIT = 85.0  # safety margin before 90s hard limit

# ---------- fast verification helpers ----------
def _make_valid_mask():
    """Boolean array of length 2**N where entry k is True iff k is a sorted 0/1 state."""
    m = np.zeros(1 << N, dtype=bool)
    for k in range(N + 1):
        if k == 0:
            val = 0
        else:
            val = (1 << N) - (1 << (N - k))
        m[val] = True
    return m

_VALID = _make_valid_mask()

def verify(network):
    """Return True if network sorts all 2**N boolean inputs (zero-one principle)."""
    if not network:
        return N <= 1
    states = np.arange(1 << N, dtype=np.uint32)
    for i, j in network:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        mask = (bi == 1) & (bj == 0)
        if mask.any():
            states[mask] ^= (1 << i) | (1 << j)
    return bool(np.all(_VALID[states]))

def compute_depth(network):
    """Greedy layering depth (reported separately; does not affect score)."""
    if not network:
        return 0
    layers = []
    for comp in network:
        placed = False
        for layer in layers:
            wires = set()
            for c in layer:
                wires.add(c[0]); wires.add(c[1])
            if comp[0] not in wires and comp[1] not in wires:
                layer.append(comp)
                placed = True
                break
        if not placed:
            layers.append([comp])
    return len(layers)

def seed_network():
    """Odd-even transposition network (136 comparators for N=17)."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# ---------- SAT-style structural search ----------
def _all_pairs():
    """Return list of all (i,j) pairs with i<j."""
    return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _layer_comparators(wires):
    """Given a list of wires, return all disjoint pairs that can be in one layer."""
    pairs = []
    used = set()
    for i in range(len(wires)):
        for j in range(i + 1, len(wires)):
            a, b = wires[i], wires[j]
            if a > b:
                a, b = b, a
            if a not in used and b not in used:
                pairs.append((a, b))
    return pairs

def _build_layered_network(num_layers, rng, layer_type='random'):
    """
    Build a layered sorting network. Each layer consists of disjoint comparators.
    layer_type: 'random' - random disjoint pairs per layer
                'adjacent' - adjacent comparators (odd-even style)
                'mixed' - mix of both
    """
    net = []
    for layer_idx in range(num_layers):
        if layer_type == 'adjacent':
            # Odd-even transposition style
            if layer_idx % 2 == 0:
                layer = [(i, i + 1) for i in range(0, N - 1, 2)]
            else:
                layer = [(i, i + 1) for i in range(1, N - 1, 2)]
        elif layer_type == 'random':
            # Random disjoint pairs
            wires = list(range(N))
            rng.shuffle(wires)
            layer = []
            for k in range(0, N - 1, 2):
                a, b = wires[k], wires[k + 1]
                if a > b:
                    a, b = b, a
                layer.append((a, b))
        else:  # mixed
            if rng.random() < 0.5:
                if layer_idx % 2 == 0:
                    layer = [(i, i + 1) for i in range(0, N - 1, 2)]
                else:
                    layer = [(i, i + 1) for i in range(1, N - 1, 2)]
            else:
                wires = list(range(N))
                rng.shuffle(wires)
                layer = []
                for k in range(0, N - 1, 2):
                    a, b = wires[k], wires[k + 1]
                    if a > b:
                        a, b = b, a
                    layer.append((a, b))
        net.extend(layer)
    return net

def _try_remove_comparator(network, idx, deadline):
    """Try to remove comparator at idx. Return new network if valid, else None."""
    if time.time() > deadline:
        return None
    new_net = network[:idx] + network[idx + 1:]
    if verify(new_net):
        return new_net
    return None

def _greedy_remove(network, deadline, rng, max_passes=10):
    """Repeatedly try to remove comparators until stuck or time runs out."""
    current = list(network)
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        indices = list(range(len(current)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            new_net = _try_remove_comparator(current, idx, deadline)
            if new_net is not None:
                current = new_net
                improved = True
                break
        if not improved:
            break
    return current

def _replace_comparator(network, idx, new_comp, deadline):
    """Replace comparator at idx with new_comp. Return new network if valid, else None."""
    if time.time() > deadline:
        return None
    new_net = network[:idx] + [new_comp] + network[idx + 1:]
    if verify(new_net):
        return new_net
    return None

def _swap_comparators(network, i, j, deadline):
    """Swap comparators at positions i and j. Return new network if valid, else None."""
    if time.time() > deadline:
        return None
    new_net = list(network)
    new_net[i], new_net[j] = new_net[j], new_net[i]
    if verify(new_net):
        return new_net
    return None

def _local_search_layered(start_net, deadline, rng, max_iterations=500):
    """
    Local search on a layered network. Moves:
    1. Replace a comparator with a random one
    2. Swap two comparators
    3. Try to remove a comparator
    Accept moves that maintain validity and reduce size.
    """
    current = list(start_net)
    best = list(current)
    best_count = len(current)
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        if len(current) <= best_count:
            # Try to improve further
            pass
        
        move_type = rng.random()
        if move_type < 0.4:
            # Replace a comparator
            if not current:
                break
            idx = rng.randint(0, len(current) - 1)
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            new_net = _replace_comparator(current, idx, (a, b), deadline)
            if new_net is not None:
                current = new_net
                current = _greedy_remove(current, deadline, rng, max_passes=3)
                if len(current) < best_count:
                    best = list(current)
                    best_count = len(current)
        elif move_type < 0.7:
            # Swap two comparators
            if len(current) < 2:
                break
            i, j = rng.sample(range(len(current)), 2)
            new_net = _swap_comparators(current, i, j, deadline)
            if new_net is not None:
                current = new_net
                current = _greedy_remove(current, deadline, rng, max_passes=3)
                if len(current) < best_count:
                    best = list(current)
                    best_count = len(current)
        else:
            # Try to remove a random comparator
            if not current:
                break
            idx = rng.randint(0, len(current) - 1)
            new_net = _try_remove_comparator(current, idx, deadline)
            if new_net is not None:
                current = new_net
                if len(current) < best_count:
                    best = list(current)
                    best_count = len(current)
    
    return best

def _construct_with_layered_search(deadline, rng):
    """
    Structural search: try different numbers of layers and layer types,
    then do local search on the best candidates.
    """
    best_net = None
    best_count = MAX_COMP
    
    # Try different numbers of layers
    for num_layers in range(8, 25):
        if time.time() > deadline:
            break
        
        for layer_type in ['adjacent', 'random', 'mixed']:
            if time.time() > deadline:
                break
            
            # Build initial layered network
            net = _build_layered_network(num_layers, rng, layer_type)
            
            # Quick check: is it valid?
            if not verify(net):
                # Try local search to make it valid
                net = _local_search_layered(net, deadline, rng, max_iterations=100)
            
            if verify(net) and len(net) < best_count:
                best_net = list(net)
                best_count = len(net)
    
    # Final local search on the best found
    if best_net is not None:
        best_net = _local_search_layered(best_net, deadline, rng, max_iterations=300)
        if verify(best_net):
            best_net = _greedy_remove(best_net, deadline, rng, max_passes=10)
    
    return best_net

def _construct_with_hill_climbing(deadline, rng):
    """
    Hill climbing from seed network with aggressive removal and replacement.
    """
    current = seed_network()
    best = list(current)
    best_count = len(current)
    
    # First, aggressive greedy removal
    current = _greedy_remove(current, deadline, rng, max_passes=20)
    if len(current) < best_count:
        best = list(current)
        best_count = len(current)
    
    # Hill climbing
    for _ in range(200):
        if time.time() > deadline:
            break
        
        move_type = rng.random()
        if move_type < 0.5:
            # Replace a comparator
            if not current:
                break
            idx = rng.randint(0, len(current) - 1)
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            new_net = _replace_comparator(current, idx, (a, b), deadline)
            if new_net is not None:
                current = new_net
                current = _greedy_remove(current, deadline, rng, max_passes=5)
                if len(current) < best_count:
                    best = list(current)
                    best_count = len(current)
        else:
            # Swap two comparators
            if len(current) < 2:
                break
            i, j = rng.sample(range(len(current)), 2)
            new_net = _swap_comparators(current, i, j, deadline)
            if new_net is not None:
                current = new_net
                current = _greedy_remove(current, deadline, rng, max_passes=5)
                if len(current) < best_count:
                    best = list(current)
                    best_count = len(current)
    
    return best

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT
    rng = random.Random(42)
    
    # Start with seed network
    best_net = seed_network()
    best_count = len(best_net)
    
    # Phase 1: Greedy removal from seed
    best_net = _greedy_remove(best_net, deadline, rng, max_passes=30)
    if verify(best_net):
        best_count = len(best_net)
    
    # Phase 2: Layered structural search
    layered_net = _construct_with_layered_search(deadline, rng)
    if layered_net is not None and verify(layered_net) and len(layered_net) < best_count:
        best_net = layered_net
        best_count = len(best_net)
    
    # Phase 3: Hill climbing from best
    hc_net = _construct_with_hill_climbing(deadline, rng)
    if hc_net is not None and verify(hc_net) and len(hc_net) < best_count:
        best_net = hc_net
        best_count = len(best_net)
    
    # Final verification
    if not verify(best_net):
        best_net = seed_network()
        best_count = len(best_net)
    
    depth = compute_depth(best_net)
    return best_net
