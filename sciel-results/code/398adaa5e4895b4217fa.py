"""Triangle-free graph without independent 10-set, maximizing n (Ramsey R(3,10))."""
import time
import random
import sys

_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def _has_triangle_circulant(S, n):
    """
    Check if a circulant graph with difference set S on Z_n is triangle-free.
    Triangle exists if there are a, b, c in S such that a+b=c, a-b=c, etc.
    Specifically, we need to ensure no x, y, z in S satisfy x+y=z (mod n) or x-y=z (mod n)
    with x,y,z distinct? No, triangle is (0, a, b) with edges (0,a), (0,b), (a,b).
    Edge (a,b) exists if a-b in S or b-a in S.
    So we need: for any a, b in S, (a-b) mod n not in S and (b-a) mod n not in S?
    Actually, if a, b in S, then 0 is connected to a and b.
    Is a connected to b? Yes if (a-b) % n in S or (b-a) % n in S.
    So we must avoid: a in S, b in S, and (a-b)%n in S.
    Note: S is usually symmetric (if d in S, n-d in S) for undirected graphs, 
    but the difference set S here represents the "generators". 
    Let S be the set of differences d such that edge (i, i+d) exists.
    For undirected, if d in S, n-d is effectively the same edge. 
    Let's define S as a set of integers in {1, ..., floor(n/2)}.
    Edge (u, v) exists if (v-u) % n in S or (u-v) % n in S.
    Triangle (0, a, b):
    Edges: (0,a) -> a in S. (0,b) -> b in S. (a,b) -> (b-a)%n in S or (a-b)%n in S.
    So we need to ensure that for all a, b in S:
    (a-b) % n not in S AND (b-a) % n not in S.
    Note that (b-a) % n = n - (a-b) % n (if a!=b).
    If S is symmetric (d in S => n-d in S), then checking (a-b)%n in S is sufficient?
    If d = (a-b)%n is in S, then n-d = (b-a)%n is also in S.
    So we just need: for all a, b in S, (a-b) % n not in S.
    Also a!=b.
    """
    for a in S:
        for b in S:
            if a == b:
                continue
            diff = (a - b) % n
            if diff in S:
                return True
    return False

def _alpha_circulant(S, n):
    """
    Calculate the independence number of a circulant graph defined by S on Z_n.
    An independent set I is a set of vertices such that no two vertices in I have a difference in S.
    This is equivalent to finding the largest subset I of Z_n such that (I-I) \ {0} is disjoint from S.
    This is a hard problem, but for n<=40 we can use branch and bound or simple DP if structure allows.
    Given the small n, a recursive search with pruning is okay, but we need it to be fast.
    
    Alternative: Use the fact that we just need to check if alpha < 10.
    We can try to construct an independent set of size 10. If we fail, alpha < 10.
    Branch and bound for Independent Set.
    """
    # Fast check: if |S| is large, alpha is small.
    # Simple upper bound: n / (avg_degree + 1) ? Not tight.
    
    # Let's use a greedy heuristic first to find a large IS.
    # If greedy finds >= 10, we might need exact check.
    # If greedy finds < 10, we still need to be sure it's not 10.
    
    # For n<=40, exact B&B is feasible if optimized.
    # Vertices 0..n-1.
    # We want max size subset I s.t. for all u,v in I, (u-v)%n not in S.
    
    S_set = set(S)
    # Also add n-S elements? No, S is defined as {1..n//2}.
    # Condition: d in S or n-d in S.
    # Let's define "conflict" relation.
    # u conflicts with v if (u-v)%n in S_set or (v-u)%n in S_set.
    
    def has_conflict(u, v):
        d = (u - v) % n
        return d in S_set or (n - d) in S_set
    
    # B&B
    best = [0]
    full_mask = (1 << n) - 1
    
    # Precompute conflict masks?
    # conflict_mask[i] = bitmask of vertices conflicting with i
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and has_conflict(i, j):
                conflict_mask[i] |= (1 << j)
                
    def bnb(available, current_size, last_added):
        if not available:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Bound: current_size + popcount(available)
        # Better bound: coloring bound on the conflict graph?
        # Or just simple: if current_size + available.bit_count() <= best[0]: return
        if current_size + available.bit_count() <= best[0]:
            return
            
        # Heuristic: pick vertex with max degree in available?
        # Or just lowest bit?
        # Let's pick lowest bit for simplicity, or max degree for better pruning.
        # Max degree in available subgraph.
        v_mask = available
        max_deg = -1
        v = -1
        while v_mask:
            lsb = v_mask & -v_mask
            u = lsb.bit_length() - 1
            v_mask ^= lsb
            deg = (conflict_mask[u] & available).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = u
                
        if v == -1:
            return
            
        # Branch 1: Include v
        # Remove v and its neighbors from available
        new_avail_include = available & ~(1 << v) & ~conflict_mask[v]
        bnb(new_avail_include, current_size + 1, v)
        
        # Branch 2: Exclude v
        new_avail_exclude = available & ~(1 << v)
        bnb(new_avail_exclude, current_size, -1)
        
    bnb(full_mask, 0, -1)
    return best[0]

def _build_edges_circulant(S, n):
    edges = []
    for u in range(n):
        for d in S:
            v = (u + d) % n
            if u < v:
                edges.append([u, v])
    return edges

def _try_circulant(n, seed):
    """
    Search for a valid difference set S for circulant graph on n vertices.
    """
    max_d = n // 2
    # We need to choose S subset of {1, ..., max_d}.
    # Constraints:
    # 1. Triangle free: No a,b in S such that (a-b)%n in S.
    # 2. Alpha < 10.
    
    # Heuristic: Start with a large S (to reduce alpha) and remove edges to break triangles?
    # Or start with small S and add?
    # Triangle constraint is very strong.
    # If we pick d1, d2, we cannot pick d1+d2 or d1-d2.
    
    # Let's use a backtracking search on S.
    # Order candidates: 1, 2, ..., max_d.
    # At each step, decide to include or exclude d.
    
    # To make it faster, we can use randomization or specific orderings.
    # Since n is small, let's try a few random permutations or just standard order with pruning.
    
    # Actually, known good sets often have structure.
    # Let's try to find ANY set.
    
    # Pruning: 
    # If we have current set S_cur, and we add d, check if triangle is formed.
    # Also, we can estimate alpha. If alpha is already < 10, we can stop adding? 
    # No, adding edges reduces alpha. We want alpha < 10.
    # So we want to ADD edges to reduce alpha, but we are constrained by triangle-free.
    # So we should try to make S as LARGE as possible subject to triangle-free, 
    # hoping that the resulting alpha is small.
    
    # So, greedy approach:
    # Candidates = {1, ..., max_d}
    # S = []
    # For d in Candidates (sorted by some heuristic):
    #   If S U {d} is triangle-free:
    #       S.append(d)
    # Check if alpha(S) < 10.
    # If not, backtrack or try different order.
    
    # Let's implement a randomized greedy with restarts.
    rng = random.Random(seed)
    
    for attempt in range(50):
        if _remaining_time() < 1:
            return None
            
        candidates = list(range(1, max_d + 1))
        rng.shuffle(candidates)
        
        S = []
        S_set = set()
        
        for d in candidates:
            # Check if adding d creates a triangle
            valid = True
            # Check against existing elements in S
            for x in S:
                diff1 = (d - x) % n
                diff2 = (x - d) % n
                if diff1 in S_set or diff2 in S_set:
                    valid = False
                    break
            # Also check if d itself conflicts? No, d != x.
            # Wait, the condition is: for all a,b in S_new, (a-b)%n not in S_new.
            # If we add d, we must check:
            # 1. For all x in S_old: (d-x)%n not in S_old and (x-d)%n not in S_old.
            #    Note: (x-d)%n = n - (d-x)%n.
            #    If S is symmetric, checking one direction might suffice?
            #    My S is not necessarily symmetric in the sense of containing n-d.
            #    But the graph is undirected, so edge (i, i+d) is same as (i+d, i).
            #    The set S defines the "positive" differences.
            #    If d in S, then n-d is the "negative" difference.
            #    The triangle condition (a-b)%n in S covers both directions if we consider the set of all differences generated by S?
            #    No, S is just the set of generators.
            #    Let's stick to: (a-b)%n in S OR (b-a)%n in S.
            #    Since (b-a)%n = n - (a-b)%n.
            #    If we maintain S as a set of integers, we just check if diff in S.
            
            if valid:
                S.append(d)
                S_set.add(d)
                
        # Check alpha
        if S:
            alpha = _alpha_circulant(S, n)
            if alpha < 10:
                return S
        else:
            # Empty set means no edges, alpha = n.
            # We need alpha < 10.
            pass

    return None

def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n.
    """
    best_S = None
    best_n = 0
    
    # Try n from 40 down to 20
    for n in range(40, 19, -1):
        if _remaining_time() < 5:
            break
            
        found = False
        # Try multiple seeds for randomized greedy
        for seed in range(200):
            if _remaining_time() < 2:
                break
                
            S = _try_circulant(n, seed)
            if S is not None:
                best_S = S
                best_n = n
                found = True
                break
                
        if found:
            break
            
    if best_S is None:
        # Fallback: small graph
        # n=20, S={1,2,3,4,5}? 
        # Let's just return a trivial valid graph if search fails completely.
        # A cycle C_n is triangle free. Alpha(C_n) ~ n/2.
        # For n=20, alpha=10. We need <10.
        # So n=19, alpha=9.
        # Let's construct C_19.
        n = 19
        best_S = [1] # Cycle edges (i, i+1)
        best_n = n
        # Verify C_19: Triangle free? Yes. Alpha(19) = floor(19/2) = 9 < 10. Valid.
        
    edges = _build_edges_circulant(best_S, best_n)
    
    return {'n': best_n, 'edges': edges}
