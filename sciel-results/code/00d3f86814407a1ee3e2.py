"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 9
SIZE = 1 << N  # 512

# Precompute neighbors for each vertex
_NEIGHBORS = []
for v in range(SIZE):
    nbrs = [v ^ (1 << b) for b in range(N)]
    _NEIGHBORS.append(nbrs)

# Precompute a fast adjacency matrix as a set of frozensets? No, use a boolean matrix for speed.
# adj_mask[v] is a bitmask of SIZE bits where bit u is 1 if u is adjacent to v.
# But SIZE=512, so we can use Python integers as bitsets (512 bits fits in a few machine words).
# Actually, for checking "non-consecutive vertices differ in at least two bits", we need to ensure
# that the new vertex v is not at Hamming distance 1 from any vertex in the path except its predecessor.
# This is equivalent to: v is not adjacent to any vertex in path[:-1].
# We can maintain a bitmask 'forbidden' that has bits set for all vertices adjacent to any vertex in the path.
# When we add a new vertex u to the path, we OR in adj_mask[u].
# When we backtrack, we need to remove it. But OR is not easily reversible.
# Alternative: maintain a count array 'adj_count[v]' = number of vertices in the path adjacent to v.
# Then v is forbidden if adj_count[v] > 0 (and v is not the predecessor).
# Actually, the condition is: v must not be adjacent to any vertex in the path EXCEPT the last one (predecessor).
# So when checking candidate v from predecessor p:
#   v is valid if v is not in used, and for all x in path[:-1], dist(v,x) >= 2.
#   dist(v,x) >= 2 is equivalent to x is not adjacent to v, i.e., adj_count_excl_pred[v] == 0.
# Let's maintain adj_count[v] = number of vertices in the entire path that are adjacent to v.
# Then for candidate v with predecessor p:
#   adj_count[v] includes p (since p is in path and adjacent to v).
#   So we need adj_count[v] - 1 == 0, i.e., adj_count[v] == 1.
#   And v not in used.

def _make_adj_mask():
    masks = [0] * SIZE
    for v in range(SIZE):
        m = 0
        for u in _NEIGHBORS[v]:
            m |= (1 << u)
        masks[v] = m
    return masks

_ADJ_MASK = _make_adj_mask()

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0  # 135s to leave margin before 150s kill
    best_path = None
    best_len = 0
    
    # We'll use Warnsdorff's rule with backtracking and randomization
    # to find long snakes.
    
    def try_snake(seed_start, max_steps=None):
        """Try to build a snake starting from seed_start using Warnsdorff backtracking."""
        if max_steps is None:
            max_steps = 5000000  # high limit, controlled by deadline
        
        path = [seed_start]
        used = [False] * SIZE
        used[seed_start] = True
        adj_count = [0] * SIZE
        # Initialize adj_count: for each neighbor of seed_start, increment
        for u in _NEIGHBORS[seed_start]:
            adj_count[u] += 1
        
        # Warnsdorff: at each step, choose the neighbor with the fewest onward options
        # Onward options for a candidate v (from predecessor p): 
        #   number of w in neighbors(v) such that w not used and adj_count[w] - 1 == 0 (i.e., adj_count[w] == 1)
        # But adj_count[w] counts all path vertices adjacent to w. Since p is in path and adjacent to v,
        # and v is candidate, w is adjacent to v. w is not adjacent to p unless w==p or dist(w,p)==1.
        # This is getting complex. Let's simplify:
        # For candidate v from p:
        #   v is valid if not used[v] and adj_count[v] == 1 (only p is adjacent to v in path).
        #   The "onward degree" of v is the number of valid next steps from v, which is:
        #   count of w in neighbors(v) such that not used[w] and adj_count[w] == 1 (after adding v).
        #   But adj_count changes when we add v. So we compute it on the fly.
        
        def get_options(v, p):
            """Get list of valid next vertices from v, given predecessor p."""
            opts = []
            for w in _NEIGHBORS[v]:
                if used[w]:
                    continue
                # w is valid if adj_count[w] == 1 (only v would be adjacent to w, since p is not adjacent to w unless w==p)
                # Wait: adj_count[w] currently counts path vertices adjacent to w.
                # If we were to add v, then adj_count[w] would increase by 1 if v is adjacent to w.
                # v is adjacent to w by construction. So after adding v, adj_count[w] becomes adj_count[w] + 1.
                # For w to be a valid next step from v, we need that after adding v, w is only adjacent to v in the path.
                # That means adj_count[w] + 1 == 1, i.e., adj_count[w] == 0.
                # But also w must not be p (since p is already used).
                if w == p:
                    continue
                if adj_count[w] == 0:
                    opts.append(w)
            return opts
        
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                return path
            
            cur = path[-1]
            # Get options from cur
            # For the first step, predecessor is none, but we handle it by checking adj_count[cur] properly.
            # Actually, let's restructure: at each step, we're at cur, and we want to find next.
            # The predecessor of cur is path[-2] if len(path) > 1, else None.
            pred = path[-2] if len(path) > 1 else None
            
            opts = []
            for w in _NEIGHBORS[cur]:
                if used[w]:
                    continue
                if w == pred:
                    continue
                # Check if w is valid: adj_count[w] should be 1 (only cur is adjacent to w in path)
                # Wait, no. adj_count[w] counts ALL path vertices adjacent to w.
                # cur is in path and adjacent to w. So adj_count[w] >= 1.
                # For w to be valid, no other path vertex should be adjacent to w.
                # So adj_count[w] == 1.
                if adj_count[w] == 1:
                    # Compute onward degree for Warnsdorff
                    onward = 0
                    for x in _NEIGHBORS[w]:
                        if x == cur or x == pred:
                            continue
                        if used[x]:
                            continue
                        if adj_count[x] == 0:
                            onward += 1
                    opts.append((onward, w))
            
            if not opts:
                break
            
            # Warnsdorff: pick the option with minimum onward degree
            opts.sort(key=lambda t: t[0])
            # Add some randomness to avoid getting stuck in the same local optimum
            if random.random() < 0.3 and len(opts) > 1:
                # Pick randomly from the top 3
                k = min(3, len(opts))
                _, nxt = random.choice(opts[:k])
            else:
                _, nxt = opts[0]
            
            # Commit to nxt
            path.append(nxt)
            used[nxt] = True
            for u in _NEIGHBORS[nxt]:
                adj_count[u] += 1
            steps += 1
        
        return path
    
    # Try multiple seeds with different randomizations
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline:
        seed = rng.randint(0, SIZE - 1)
        path = try_snake(seed)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path
        attempts += 1
        # Also try to improve by splicing? For now, just multiple seeds.
        # Limit total attempts to avoid infinite loop if deadline check is slow
        if attempts > 10000:
            break
    
    if best_path is None:
        # Fallback to simple greedy
        path, used = [0], {0}
        while True:
            cur, nxt = path[-1], None
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used or any((v ^ p).bit_count() == 1 for p in path[:-1]):
                    continue
                nxt = v
                break
            if nxt is None:
                return path
            path.append(nxt)
            used.add(nxt)
        return path
    
    return best_path
# EVOLVE-BLOCK-END
