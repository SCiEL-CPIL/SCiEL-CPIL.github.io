"""Cap set greedy builder with conflict-graph degree minimization."""
import time
import numpy as np

_start_time = None
_DEADLINE = 330.0 * 0.95  # 313.5s safety margin

# Cache for conflict graph and state
_cache = {}

def _build_conflict_graph(n):
    """
    Build the conflict graph on {0,1,2}^n.
    Two vectors conflict if they are endpoints of a 3-term AP in F_3^n,
    i.e., there exists a third vector w such that v + w = 2*u mod 3
    for some u, meaning v, u, w form a line.
    
    Two vectors v, w conflict (can't both be in cap set) if:
    For all coordinates i: v[i] == w[i] OR v[i] != w[i]
    Actually, v and w form a 2-point "potential AP" if there exists
    a direction d such that v + d and v + 2d (mod 3) are both valid,
    and w is one of those. More precisely:
    
    v and w are in conflict if there exists a vector u such that
    {v, u, w} forms a 3-term AP in F_3^n.
    
    A 3-term AP in F_3^n: v, v+d, v+2d where d != 0.
    So v and w conflict iff w = v + 2d for some nonzero d,
    i.e., w - v = 2d mod 3, i.e., w - v is any nonzero vector (since 2 is invertible mod 3).
    
    Wait, that means ANY two distinct vectors conflict? No, that's not right.
    
    Let me reconsider. A cap set has no 3-term AP. Two vectors v, w can both be in
    a cap set as long as there's no u such that v, u, w form a 3-term AP.
    
    v, u, w form a 3-term AP iff u = (v+w)/2 mod 3. Since 2^{-1} = 2 mod 3,
    u = 2*(v+w) mod 3 = (-1)*(v+w) mod 3 = (6-v-w) mod 3... 
    
    Actually in F_3: 2 * (v+w) mod 3. Since 2 ≡ -1 mod 3, u_i = 2*(v_i + w_i) mod 3.
    
    But u must also be in {0,1,2}^n, which it always is. So ANY two distinct vectors
    v, w have a midpoint u = 2*(v+w) mod 3. So any two vectors can form a 3-term AP!
    
    That means the conflict graph is complete? That can't be right because cap sets exist.
    
    Oh wait, I see the issue. The midpoint u = 2*(v+w) mod 3 is always valid. But the
    3-term AP is (v, u, w) where u must be the midpoint. The constraint is that NO three
    vectors in the set form a 3-term AP. So when adding w, we need to check that for every
    pair (v1, v2) already in the set, the midpoint m = 2*(v1+v2) mod 3 is NOT equal to w,
    AND w is not the midpoint of any existing pair, AND no existing vector is the midpoint
    of w and another existing vector.
    
    Actually the correct check: adding w to set S creates a 3-term AP iff:
    1. There exist v1, v2 in S such that w = 2*(v1+v2) mod 3 (w is midpoint of v1,v2)
    2. There exists v in S such that v = 2*(w+v') mod 3 for some v' in S (v is midpoint of w,v')
       which means w+v' = 2*v mod 3, i.e., v' = 2*v - w mod 3... 
    
    Let me think more carefully. A 3-term AP is (a, b, c) where b = (a+c)/2 = 2*(a+c) mod 3.
    So given two endpoints a, c, the midpoint is b = 2*(a+c) mod 3.
    
    When adding w to S, a new 3-term AP is formed iff:
    - w is the midpoint of two existing vectors: w = 2*(v1+v2) mod 3 for some v1,v2 in S
    - w is an endpoint: for some v in S, the other endpoint c satisfies v = 2*(w+c) mod 3,
      so c = 2*(v - w) mod 3... wait: v = 2*(w+c) mod 3 => w+c = 2^{-1}*v = 2*v mod 3 => c = 2*v - w mod 3.
      So if c = 2*v - w mod 3 is in S, then (w, v, c) or (c, v, w) forms an AP with v as midpoint.
    
    So the conflict degree of w given set S is:
    degree(w, S) = |{v in S : 2*v - w mod 3 is also in S}| + |{(v1,v2) in S x S : v1 != v2, w = 2*(v1+v2) mod 3}|
    
    For efficiency, I'll precompute for each w, the set of v such that 2*v - w is also a valid vector,
    and use a bitset of S to check membership.
    
    Actually, for the greedy algorithm, we want to pick w that minimizes conflicts with S.
    The "conflict degree" of w relative to S is the number of pairs in S that would form
    a 3-term AP with w.
    
    Let me define: for each w, conflict_count(w, S) = number of v in S such that (2*v - w) mod 3 is also in S.
    This counts how many existing vectors v have their "partner" (the vector that would make
    (w, v, partner) a 3-term AP) also in S.
    
    We want to minimize this.
    """
    total = 3 ** n
    # For each vector w (indexed 0..total-1), precompute the mapping:
    # For each v, partner = (2*v - w) mod 3
    # We'll store this as a numpy array of shape (total, total) would be too large for large n.
    # Instead, we compute on the fly using the vector representation.
    
    # Convert all vectors to base-3 indices
    # index = v[0] * 3^(n-1) + v[1] * 3^(n-2) + ... + v[n-1]
    powers = 3 ** np.arange(n - 1, -1, -1)
    
    # Build array of all vectors
    all_vectors = np.array([list(map(int, np.array3base(i, n))) for i in range(total)])
    # This is slow. Let's use a different approach.
    
    # Actually, let's just compute the partner on the fly using numpy.
    # For a given w and the set S (as a boolean array), compute conflict count.
    
    return total, powers

def _vec_to_idx(vec, n):
    """Convert vector to base-3 index."""
    idx = 0
    for i, v in enumerate(vec):
        idx += v * (3 ** (n - 1 - i))
    return idx

def priority(vector, n):
    """
    Dynamic priority based on conflict-graph degree minimization.
    
    The idea: among all candidate vectors, prefer those with the fewest
    potential conflicts with the already-selected set. This is modeled as
    a graph where edges represent 3-term AP relationships, and we greedily
    select the vertex with minimum degree in the induced subgraph.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        _cache['state'] = {'n': n, 'selected': set(), 'total': 3 ** n, 'powers': 3 ** np.arange(n - 1, -1, -1)}
    
    if time.time() - _start_time > _DEADLINE:
        return 0.0
    
    state = _cache['state']
    if state['n'] != n:
        state['n'] = n
        state['selected'] = set()
        state['total'] = 3 ** n
        state['powers'] = 3 ** np.arange(n - 1, -1, -1)
    
    # Convert vector to index
    idx = _vec_to_idx(vector, n)
    
    # If already selected, give it very low priority
    if idx in state['selected']:
        return -1e10
    
    # Compute conflict count: number of v in selected such that partner = (2*v - vector) mod 3 is also in selected
    selected = state['selected']
    conflict_count = 0
    
    # For each selected vector v, compute partner = (2*v - w) mod 3
    # If partner is also in selected, that's a conflict.
    # To speed up, we can precompute for the current w, which v's have partners in selected.
    
    # Actually, for efficiency with large sets, let's use a different approach:
    # Precompute for each possible w, the set of v's that would conflict.
    # But that's 3^n * 3^n which is too large.
    
    # Alternative: For the current w, iterate over selected and check.
    # This is O(|S|) per query, which might be slow for large S.
    
    # Let's use a smarter approach: precompute the "partner mapping" for w.
    # For a fixed w, partner(v) = (2*v - w) mod 3 component-wise.
    # We need to count how many v in S have partner(v) also in S.
    
    # Convert w to numpy array
    w = np.array(vector, dtype=np.int8)
    
    # For each v in selected, compute partner and check if in selected
    # To make this faster, convert selected to a set of tuples for O(1) lookup
    selected_set = selected  # already a set of indices
    
    # Convert w to base-3 digits
    w_digits = np.zeros(n, dtype=np.int8)
    tmp = idx
    for i in range(n - 1, -1, -1):
        w_digits[i] = tmp % 3
        tmp //= 3
    
    conflict = 0
    for v_idx in selected_set:
        # Compute v digits
        v_digits = np.zeros(n, dtype=np.int8)
        tmp = v_idx
        for i in range(n - 1, -1, -1):
            v_digits[i] = tmp % 3
            tmp //= 3
        
        # partner = (2*v - w) mod 3
        partner_digits = (2 * v_digits - w_digits) % 3
        
        # Convert partner to index
        partner_idx = 0
        for i in range(n):
            partner_idx += partner_digits[i] * (3 ** (n - 1 - i))
        
        if partner_idx in selected_set:
            conflict += 1
    
    # Primary: minimize conflict count
    # Secondary: among equal conflicts, prefer vectors with more distinct symbols
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    distinct = (c0 > 0) + (c1 > 0) + (c2 > 0)
    
    # Tertiary: balance
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # Quaternary: hash for tie-breaking
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF
    
    # Combine: conflict is dominant (minimize), then distinct (maximize), then balance (minimize dev)
    score = -conflict * 100000.0 + distinct * 1000.0 - dev * 10.0 + hash_score
    
    return float(score)
