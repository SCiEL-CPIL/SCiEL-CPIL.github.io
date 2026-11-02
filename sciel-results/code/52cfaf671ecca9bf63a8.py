"""
Third autocorrelation inequality: minimize C3 = max_k |c_k| for a signed step
function with 400 equal intervals on [-1/4, 1/4].

c_k = D * sum_j h_j h_{j+k}, D = 1/100.
Minimizing max_k |c_k| <=> minimizing max_k |(h*h)_k| since D is constant.
h_j in {-1, +1}.

Structural change: Represent the sign sequence as the output of a 1D elementary
cellular automaton (ECA) applied to a random seed. We evolve the CA for T steps
and map the final state (or a subset) to the 400-bit sign vector. This generates
long-range dependent structures that are difficult for local coordinate-flip
search to produce or escape from, potentially bypassing local minima traps.

We combine CA-generated seeds with coordinate-flip local search to refine the
solution. Multiple CA rules and random seeds are tried, and the best result is
returned.

Deadline guard: explicit wall-clock tracking with safety margin.
"""
import time
import numpy as np

N = 400
DEADLINE = 150.0
MARGIN = 8.0
RNG_SEED = 42


def _max_abs_conv(h):
    """Exact max over lags of |(h*h)_k| for h in {+-1}."""
    c = np.convolve(h, h, mode='full')
    return int(np.max(np.abs(c)))


def _eca_step(state, rule):
    """Apply one step of an elementary cellular automaton (rule 0-255)."""
    n = len(state)
    new_state = np.empty_like(state)
    # Wrap-around boundary conditions
    left = np.roll(state, 1)
    right = np.roll(state, -1)
    # Compute index for rule lookup: (left, center, right) as 3-bit number
    idx = (left << 2) | (state << 1) | right
    # Rule is a 256-bit integer; bit i of rule determines output for pattern i
    rule_bits = ((rule >> np.arange(8)) & 1).astype(np.int8)
    # Actually, rule is 8 bits for 8 patterns. Let's use a lookup table.
    # Better approach: precompute lookup for all 8 patterns.
    # But for speed, let's just use the standard ECA definition.
    # Rule r: for pattern p (0-7), output is (r >> p) & 1
    # But we need to map idx (0-7) to output.
    # Let's build a small lookup table once per rule.
    # For simplicity and speed, use numpy operations.
    # idx is 0..7, so we can use a lookup.
    lut = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.int8)
    new_state = lut[idx]
    return new_state


def _eca_evolve(seed_state, rule, steps):
    """Evolve ECA for given steps, return final state."""
    state = seed_state.copy()
    for _ in range(steps):
        state = _eca_step(state, rule)
    return state


def _ca_to_sign_vector(ca_state, N):
    """Map CA state to a sign vector of length N."""
    # If CA state is longer than N, take first N; if shorter, repeat/truncate
    if len(ca_state) >= N:
        return ca_state[:N].astype(np.int64)
    else:
        # Tile to reach N
        reps = (N + len(ca_state) - 1) // len(ca_state)
        return np.tile(ca_state, reps)[:N].astype(np.int64)


def _local_search(h, c, limit):
    """Coordinate-flip local search with incremental updates."""
    L = 2 * N - 1
    cur = int(np.max(np.abs(c)))
    rng = np.random.default_rng(int(time.time() * 1000) % (2**31))
    
    while time.time() < limit:
        improved = False
        order = rng.permutation(N)
        for i in order:
            lo = i - (N - 1)
            hi = i + (N - 1)
            lo = max(0, lo)
            hi = min(L - 1, hi)
            if lo > hi:
                continue
            jlo = lo - i
            jhi = hi - i
            delta = 2 * int(h[i]) * h[jlo:jhi + 1]
            new_c = c[lo:hi + 1] + delta
            new_max = int(np.max(np.abs(new_c)))
            if new_max < cur:
                c[lo:hi + 1] = new_c
                h[i] = -h[i]
                cur = new_max
                improved = True
        if not improved:
            break
    
    return h, cur


def _construct():
    t0 = time.time()
    limit = DEADLINE - MARGIN
    rng = np.random.default_rng(RNG_SEED)
    
    best_h = None
    best_val = None
    
    # Try several ECA rules and random seeds
    # ECA rules known for interesting behavior: 30, 45, 54, 110, 126, 150, 184, 225
    ca_rules = [30, 45, 54, 110, 126, 150, 184, 225, 73, 86, 90, 105, 122, 149, 162, 183]
    
    # Also include a pure random start as baseline
    for _ in range(3):
        h = rng.choice(np.array([-1, 1], dtype=np.int64), size=N)
        c = np.convolve(h, h, mode='full')
        v = int(np.max(np.abs(c)))
        if best_val is None or v < best_val:
            best_val, best_h = v, h.copy()
        if time.time() > limit:
            break
    
    # Try CA-generated seeds
    ca_state_len = N  # Use same length as N
    for rule in ca_rules:
        if time.time() > limit:
            break
        # Try a few random seeds for this rule
        for seed_iter in range(5):
            if time.time() > limit:
                break
            seed_state = rng.integers(0, 2, size=ca_state_len).astype(np.int8)
            # Evolve for a few steps
            steps = rng.integers(1, 10)
            final_state = _eca_evolve(seed_state, rule, steps)
            h = _ca_to_sign_vector(final_state, N)
            
            # Ensure h is in {-1, 1}
            h = 2 * h - 1  # 0 -> -1, 1 -> +1
            
            c = np.convolve(h, h, mode='full')
            v = int(np.max(np.abs(c)))
            
            # Refine with local search
            h_ref, v_ref = _local_search(h.copy(), c.copy(), limit)
            
            if best_val is None or v_ref < best_val:
                best_val, best_h = v_ref, h_ref.copy()
            
            if time.time() > limit:
                break
    
    # Use remaining time for more local search from the best solution
    if best_h is not None and time.time() < limit:
        c = np.convolve(best_h, best_h, mode='full')
        best_h, best_val = _local_search(best_h.copy(), c.copy(), limit)
    
    return best_h


def construct():
    return _construct()


# ---- local self-test (not called by evaluator) ----
if __name__ == '__main__':
    t = time.time()
    h = construct()
    print("time", time.time() - t, "N", len(h), "max_abs_conv", _max_abs_conv(h))
