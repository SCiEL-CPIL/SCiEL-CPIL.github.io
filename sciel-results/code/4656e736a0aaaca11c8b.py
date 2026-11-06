"""Size-efficient sorting networks on 17 wires – improved search."""
import time
import random
import numpy as np

N = 17
MAX_COMP = 4096

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
    states = np.arange(1 << N, dtype=np.uint32)
    for i, j in network:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        mask = (bi == 1) & (bj == 0)
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

# ---------- search helpers ----------
def try_remove(network, idx):
    """Return new network if removing comparator idx keeps validity, else None."""
    new_net = network[:idx] + network[idx + 1:]
    if verify(new_net):
        return new_net
    return None

def greedy_remove_until_fixed(network, deadline, rng, verbose=False):
    """Repeatedly try to remove any comparator (random order each pass) until stuck."""
    current = list(network)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(current)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() >= deadline:
                break
            reduced = try_remove(current, idx)
            if reduced is not None:
                current = reduced
                improved = True
                if verbose:
                    print(f"  removed one -> {len(current)}", flush=True)
                break
    return current

def build_batch_network(rng, n_comps, max_wires_per_step=4):
    """
    Build a random but structured sorting network by composing random
    permutation-like layers (sets of disjoint comparators).
    This tends to produce networks closer to valid than pure random.
    """
    net = []
    wires = list(range(N))
    # We'll create layers; each layer is a set of disjoint comparators.
    # Number of comparators per layer: up to floor(N/2)=8, but we limit for diversity.
    while len(net) < n_comps:
        # Choose a random subset of wires to pair in this layer
        available = wires[:]
        rng.shuffle(available)
        layer = []
        for i in range(0, len(available), 2):
            if i + 1 >= len(available):
                break
            a, b = available[i], available[i + 1]
            if a > b:
                a, b = b, a
            layer.append((a, b))
        if not layer:
            # Try a different random pairing
            rng.shuffle(wires)
            continue
        # Interleave with some fixed adjacent comparators to help sorting
        if rng.random() < 0.5:
            adj = [(k, k + 1) for k in range(0, N - 1, 2)]
            layer = layer + adj
        # Trim to avoid exceeding budget
        room = n_comps - len(net)
        if len(layer) > room:
            layer = layer[:room]
        net.extend(layer)
    return net[:n_comps]

def local_repair(network, deadline, rng, target_count=None):
    """
    Local search: try random replacements and reorders, then greedy remove.
    Accept a move only if it keeps validity (or improves count after removal).
    """
    current = list(network)
    best = list(current)
    best_count = len(current)
    target = target_count if target_count is not None else best_count

    for _ in range(200):  # iterations within this repair
        if time.time() >= deadline:
            break
        if len(current) <= target:
            break

        move = rng.random()
        if move < 0.5:
            # Replace a random comparator
            if not current:
                break
            idx = rng.randint(0, len(current) - 1)
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            new_net = current[:idx] + [(a, b)] + current[idx + 1:]
            if verify(new_net):
                current = new_net
                # Try to remove
                current = greedy_remove_until_fixed(current, deadline, rng)
                if len(current) < best_count:
                    best = list(current)
                    best_count = len(current)
                    if verbose:
                        print(f"    repair: {best_count}", flush=True)
        elif move < 0.8:
            # Swap two comparators
            if len(current) < 2:
                break
            i, j = rng.sample(range(len(current)), 2)
            new_net = list(current)
            new_net[i], new_net[j] = new_net[j], new_net[i]
            if verify(new_net):
                current = new_net
                current = greedy_remove_until_fixed(current, deadline, rng)
                if len(current) < best_count:
                    best = list(current)
                    best_count = len(current)
        else:
            # Insert a random comparator at random position
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            pos = rng.randint(0, len(current))
            new_net = current[:pos] + [(a, b)] + current[pos:]
            if verify(new_net):
                current = new_net
                current = greedy_remove_until_fixed(current, deadline, rng)
                if len(current) < best_count:
                    best = list(current)
                    best_count = len(current)

    return best

# ---------- main construct ----------
def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 60s hard limit
    rng = random.Random(2024)

    best_net = seed_network()
    best_count = len(best_net)
    print(f"Seed network: {best_count} comparators", flush=True)

    # Ensure seed is valid (it is, but be safe)
    if not verify(best_net):
        # Extremely unlikely; rebuild
        best_net = []
        best_count = 0

    # Phase 1: Aggressive greedy removal from seed
    print("Phase 1: Greedy removal from seed...", flush=True)
    for restart in range(30):
        if time.time() >= deadline:
            break
        cand = list(best_net)
        cand = greedy_remove_until_fixed(cand, deadline, rng, verbose=False)
        if len(cand) < best_count:
            best_net = cand
            best_count = len(cand)
            print(f"  Phase1 restart {restart}: improved to {best_count}", flush=True)

    # Phase 2: Random batch construction + local repair
    # Try various sizes around current best and smaller
    print("Phase 2: Random batch + local repair...", flush=True)
    sizes_to_try = []
    # Try sizes from best_count down to 50, stepping by 3, plus a few above
    for s in range(max(50, best_count - 20), best_count + 5, 2):
        sizes_to_try.append(s)
    # Also try some fixed known-good sizes
    for s in [71, 70, 69, 68, 67, 66, 65, 64, 63]:
        if s not in sizes_to_try and s < best_count:
            sizes_to_try.append(s)

    for size in sizes_to_try:
        if time.time() >= deadline:
            break
        if size >= best_count:
            continue
        # Try several random constructions of this size
        for attempt in range(5):
            if time.time() >= deadline:
                break
            cand = build_batch_network(rng, size)
            # First, see if it's even close to valid by a quick partial check
            # We'll just verify fully; if invalid, local repair may fix it
            if not verify(cand):
                # Try local repair to make it valid and reduce
                cand = local_repair(cand, deadline, rng, target_count=size)
            else:
                # Already valid; try to reduce further
                cand = greedy_remove_until_fixed(cand, deadline, rng)

            if verify(cand) and len(cand) < best_count:
                best_net = cand
                best_count = len(cand)
                print(f"  Phase2 size {size} attempt {attempt}: improved to {best_count}", flush=True)

    # Phase 3: Fine-grained local search on the best found
    print("Phase 3: Fine local search...", flush=True)
    for restart in range(50):
        if time.time() >= deadline:
            break
        cand = list(best_net)
        # Small random perturbations
        for _ in range(10):
            if time.time() >= deadline:
                break
            if not cand:
                break
            idx = rng.randint(0, len(cand) - 1)
            a, b = rng.sample(range(N), 2)
            if a > b:
                a, b = b, a
            new_net = cand[:idx] + [(a, b)] + cand[idx + 1:]
            if verify(new_net):
                cand = new_net
                cand = greedy_remove_until_fixed(cand, deadline, rng)
                if len(cand) < best_count:
                    best_net = cand
                    best_count = len(cand)
                    print(f"  Phase3 restart {restart}: improved to {best_count}", flush=True)
                    break

    # Final verification
    if not verify(best_net):
        best_net = seed_network()
        best_count = len(best_net)

    depth = compute_depth(best_net)
    print(f"Final: {best_count} comparators, depth {depth}", flush=True)
    return best_net
