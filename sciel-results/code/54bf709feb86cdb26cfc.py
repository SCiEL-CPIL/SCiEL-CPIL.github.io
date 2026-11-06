import numpy as np
import time

N = 17
MAX_COMPARATORS = 4096
_NUM_INPUTS = 1 << N
_DEADLINE_OFFSET = 78.0  # 90s limit with 12s safety margin

def _all_inputs():
    """Return a (2^N, N) uint8 matrix with all Boolean inputs."""
    idx = np.arange(_NUM_INPUTS, dtype=np.uint32)
    out = np.empty((_NUM_INPUTS, N), dtype=np.uint8)
    for k in range(N):
        out[:, k] = (idx >> (N - 1 - k)) & 1
    return out

def construct():
    t0 = time.monotonic()
    deadline = t0 + _DEADLINE_OFFSET

    # Precompute inputs
    inputs = _all_inputs()

    # --- Simulation helpers ---
    def apply_layer(state, layer):
        """Apply a layer of disjoint comparators in place."""
        for i, j in layer:
            si = state[:, i]
            sj = state[:, j]
            m = si < sj
            state[:, i] = np.where(m, si, sj)
            state[:, j] = np.where(m, sj, si)

    def simulate(state, layers):
        """Simulate full network on state (list of layers)."""
        for layer in layers:
            apply_layer(state, layer)

    def is_valid_network(layers):
        """Check if network sorts all inputs."""
        state = inputs.copy()
        simulate(state, layers)
        return np.all(state[:, 1:] >= state[:, :-1])

    def count_comps(layers):
        return sum(len(l) for l in layers)

    # --- OETN generation ---
    def generate_oetn():
        layers = []
        for phase in range(N):
            layer = []
            for i in range(phase % 2, N - 1, 2):
                layer.append((i, i + 1))
            if layer:
                layers.append(layer)
        return layers

    def layer_is_disjoint(layer):
        used = set()
        for i, j in layer:
            if i in used or j in used:
                return False
            used.add(i)
            used.add(j)
        return True

    # --- Greedy removal: try removing each comparator in sequence ---
    def greedy_remove(layers, t_deadline):
        """Repeatedly try removing each comparator; keep going until no more can be removed."""
        improved = True
        while improved and time.monotonic() < t_deadline:
            improved = False
            n_layers = len(layers)
            for li in range(n_layers):
                if time.monotonic() >= t_deadline:
                    break
                if not layers[li]:
                    continue
                for ci in range(len(layers[li])):
                    if time.monotonic() >= t_deadline:
                        break
                    # Remove comparator at position
                    comp = layers[li][ci]
                    new_layer = layers[li][:ci] + layers[li][ci+1:]
                    candidate = [l[:] for l in layers]
                    candidate[li] = new_layer
                    # Remove empty layers
                    candidate = [l for l in candidate if l]
                    if not candidate:
                        continue
                    if is_valid_network(candidate):
                        layers = candidate
                        improved = True
                        break
                if improved:
                    break
        return layers

    # --- Swap-based local search ---
    def try_swaps(layers, t_deadline, max_iters=2000):
        """Try swapping pairs of comparators between layers or positions."""
        import random
        rng = random.Random(42)
        cur = [l[:] for l in layers]
        cur_count = count_comps(cur)
        
        for _ in range(max_iters):
            if time.monotonic() >= t_deadline:
                break
            
            # Pick two random comparators from the network
            all_comps = []
            for li, layer in enumerate(cur):
                for ci, comp in enumerate(layer):
                    all_comps.append((li, ci, comp))
            
            if len(all_comps) < 2:
                break
            
            (l1, c1, comp1), (l2, c2, comp2) = rng.sample(all_comps, 2)
            if l1 == l2 and c1 == c2:
                continue
            
            # Try swapping comp1 and comp2 positions
            candidate = [l[:] for l in cur]
            # Remove comp1 from its position
            candidate[l1].pop(c1)
            # Remove comp2 from its position (adjust index if same layer and c2 < c1)
            if l1 == l2 and c2 > c1:
                c2 -= 1
            candidate[l2].pop(c2)
            # Insert comp1 at position c2 in layer l2
            candidate[l2].insert(min(c2, len(candidate[l2])), comp1)
            # Insert comp2 at position c1 in layer l1
            candidate[l1].insert(min(c1, len(candidate[l1])), comp2)
            
            # Check layer validity
            ok = True
            for layer in candidate:
                if not layer_is_disjoint(layer):
                    ok = False
                    break
            if not ok:
                continue
            
            # Remove empty layers
            candidate = [l for l in candidate if l]
            if not candidate:
                continue
            
            if is_valid_network(candidate):
                cur = candidate
                cur_count = count_comps(cur)
        
        return cur

    # --- Deletion with position reordering (move comparator to different layer) ---
    def try_moves(layers, t_deadline, max_iters=1000):
        """Try moving comparators to different layers."""
        import random
        rng = random.Random(123)
        cur = [l[:] for l in layers]
        
        for _ in range(max_iters):
            if time.monotonic() >= t_deadline:
                break
            
            n_layers = len(cur)
            if n_layers == 0:
                break
            
            li = rng.randint(0, n_layers - 1)
            if not cur[li]:
                continue
            ci = rng.randint(0, len(cur[li]) - 1)
            comp = cur[li][ci]
            
            target_li = rng.randint(0, n_layers - 1)
            if target_li == li:
                continue
            
            candidate = [l[:] for l in cur]
            candidate[li].pop(ci)
            candidate[target_li].append(comp)
            
            if not layer_is_disjoint(candidate[target_li]):
                continue
            
            candidate = [l for l in candidate if l]
            if not candidate:
                continue
            
            if is_valid_network(candidate):
                cur = candidate
        
        return cur

    # --- Main search ---
    # Start from OETN
    layers = generate_oetn()
    
    # Phase 1: Aggressive greedy removal
    layers = greedy_remove(layers, deadline - 30.0)
    
    # Phase 2: Try moves to rearrange
    if time.monotonic() < deadline - 20.0:
        layers = try_moves(layers, deadline - 20.0, max_iters=500)
        # Remove again after moves
        layers = greedy_remove(layers, deadline - 10.0)
    
    # Phase 3: Swap-based local search
    if time.monotonic() < deadline - 5.0:
        layers = try_swaps(layers, deadline - 5.0, max_iters=1000)
        layers = greedy_remove(layers, deadline - 2.0)
    
    # Final verification
    if not is_valid_network(layers):
        layers = generate_oetn()
    
    # Flatten
    result = []
    for layer in layers:
        for i, j in layer:
            result.append((i, j))
    
    if len(result) > MAX_COMPARATORS:
        result = result[:MAX_COMPARATORS]
    
    return result
