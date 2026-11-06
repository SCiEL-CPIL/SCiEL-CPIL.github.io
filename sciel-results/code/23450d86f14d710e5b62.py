"""Optimized constructive approach for minimum maximal no-isosceles set."""
import time
import random
import numpy as np
from itertools import product

N = 16
V = N * N  # 256 grid points

def _precompute():
    """Precompute squared distances and pair-level isosceles conflicts."""
    points = list(product(range(N), repeat=2))
    pts = np.array(points, dtype=np.int32)
    
    # Squared distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2)
    
    # For each pair (i, j), compute the set of k such that {i,j,k} is isosceles
    # Store as a bitset (array of 4 uint64 words for 256 bits)
    # pair_conflicts[i][j] = bitset of points k that conflict with pair (i,j)
    pair_conflicts = np.zeros((V, V, 4), dtype=np.uint64)
    
    for i in range(V):
        for j in range(i + 1, V):
            dij = dist[i, j]
            # k where d(i,k) == dij (k != j)
            mask1 = (dist[i] == dij).astype(bool)
            mask1[j] = False
            # k where d(j,k) == dij (k != i)  
            mask2 = (dist[j] == dij).astype(bool)
            mask2[i] = False
            # k where d(i,k) == d(j,k) (perpendicular bisector, k != i, k != j)
            mask3 = (dist[i] == dist[j]).astype(bool)
            mask3[i] = False
            mask3[j] = False
            
            combined = mask1 | mask2 | mask3
            # Convert to bitset
            ks = np.where(combined)[0]
            for k in ks:
                word_idx = k // 64
                bit_idx = k % 64
                pair_conflicts[i, j, word_idx] |= (1 << bit_idx)
                pair_conflicts[j, i, word_idx] |= (1 << bit_idx)
    
    return points, dist, pair_conflicts

def _bitset_to_set(bitset):
    """Convert a 4-word uint64 bitset to a set of indices."""
    result = set()
    for w in range(4):
        word = bitset[w]
        while word:
            lsb = word & (-word)
            bit = int(np.log2(lsb)) if lsb > 0 else 0
            if lsb > 0:
                result.add(w * 64 + bit)
            word ^= lsb
    return result

def _set_to_bitset(s):
    """Convert a set of indices to a 4-word uint64 bitset."""
    bitset = np.zeros(4, dtype=np.uint64)
    for i in s:
        bitset[i // 64] |= (1 << (i % 64))
    return bitset

def _bitset_count(bitset):
    """Count set bits in a 4-word bitset."""
    count = 0
    for w in range(4):
        word = bitset[w]
        while word:
            count += 1
            word &= word - 1
    return count

def _bitset_intersection(a, b):
    """Bitwise AND of two 4-word bitsets."""
    return a & b

def _bitset_union(a, b):
    """Bitwise OR of two 4-word bitsets."""
    return a | b

def _is_valid_set(S, dist):
    """Check if S contains no isosceles triangle."""
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            for k in S:
                if k != a and k != b:
                    dab = dist[a, b]
                    dac = dist[a, k]
                    dbc = dist[b, k]
                    if dab == dac or dab == dbc or dac == dbc:
                        return False
    return True

def _is_maximal(S, dist, pair_conflicts):
    """Check if S is maximal: no point outside S can be added without creating isosceles."""
    S_set = set(S)
    for p in range(V):
        if p in S_set:
            continue
        # Check if p can be added: for all pairs (a,b) in S, {a,b,p} must be isosceles
        # i.e., p must be in pair_conflicts[a][b] for ALL pairs (a,b) in S
        can_add = True
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                a, b = S[i], S[j]
                # Check if p is in pair_conflicts[a][b]
                word_idx = p // 64
                bit_idx = p % 64
                if not (pair_conflicts[a, b, word_idx] & (1 << bit_idx)):
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            return False
    return True

def _get_blocked_by_S(S, pair_conflicts):
    """Get the set of all points blocked by S (points that conflict with at least one pair in S)."""
    # A point p is blocked if it is in pair_conflicts[a][b] for at least one pair (a,b) in S
    # We want points that are NOT blocked (can be added)
    # Actually for maximality: p is blocked if for ALL pairs (a,b) in S, p is in pair_conflicts[a][b]
    # Wait, let me re-read: "no further grid point can be added"
    # Adding p creates isosceles if there exists a pair (a,b) in S such that {a,b,p} is isosceles
    # So p is blocked if EXISTS a pair (a,b) in S with p in pair_conflicts[a][b]
    # For maximality: for every p not in S, p must be blocked
    # i.e., for every p not in S, EXISTS (a,b) in S such that p in pair_conflicts[a][b]
    
    blocked = np.zeros(4, dtype=np.uint64)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            blocked = _bitset_union(blocked, pair_conflicts[a, b])
    return blocked

def _construct_bnb(dist, points, pair_conflicts, time_budget=280.0):
    """
    Branch and bound to find minimum maximal no-isosceles set.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_size = V + 1
    best_set = None
    
    # Precompute: for each point p, which pairs (a,b) have p in their conflict set
    # This helps with lower bound computation
    
    # Strategy: 
    # 1. Start with a greedy solution to get an upper bound
    # 2. Use B&B to try to improve
    
    # First, get a good initial solution via greedy
    def _greedy_init(rng):
        S = []
        U = set(range(V))
        
        while U:
            # Find candidates that can be added to S
            candidates = []
            for p in U:
                can_add = True
                for a in S:
                    for b in S:
                        if a >= b:
                            continue
                        # Check if {a,b,p} is isosceles
                        dab = dist[a, b]
                        dap = dist[a, p]
                        dbp = dist[b, p]
                        if dab == dap or dab == dbp or dap == dbp:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    candidates.append(p)
            
            if not candidates:
                break
            
            # Score: how many points in U would be blocked
            best_score = -1
            best_p = None
            best_blocked = set()
            
            cand_list = candidates[:]
            rng.shuffle(cand_list)
            
            for p in cand_list:
                new_blocked = set()
                for a in S:
                    dab = dist[a, p]
                    for u in U:
                        if u == p or u in new_blocked:
                            continue
                        da = dist[a, u]
                        dp = dist[p, u]
                        if dab == da or dab == dp or da == dp:
                            new_blocked.add(u)
                
                score = len(new_blocked)
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_blocked = new_blocked
                elif score == best_score and rng.random() < 0.2:
                    best_p = p
                    best_blocked = new_blocked
            
            if best_p is None:
                break
            
            S.append(best_p)
            U.remove(best_p)
            U -= best_blocked
        
        return S
    
    # Get initial solution
    rng = random.Random(42)
    initial_S = _greedy_init(rng)
    
    if not initial_S:
        return []
    
    # Verify and get upper bound
    if _is_valid_set(initial_S, dist) and _is_maximal(initial_S, dist, pair_conflicts):
        best_size = len(initial_S)
        best_set = list(initial_S)
    
    # Try multiple random restarts for better initial solution
    for trial in range(20):
        if time.time() > deadline:
            break
        rng = random.Random(trial * 104729 + 7)
        S = _greedy_init(rng)
        if S and _is_valid_set(S, dist) and _is_maximal(S, dist, pair_conflicts):
            if len(S) < best_size:
                best_size = len(S)
                best_set = list(S)
    
    # Local search: try to remove points while maintaining validity and maximality
    def _local_search(S, time_limit=5.0):
        start_ls = time.time()
        deadline_ls = start_ls + time_limit
        S = list(S)
        improved = True
        while improved and time.time() < deadline_ls:
            improved = False
            for idx in range(len(S)):
                if time.time() > deadline_ls:
                    break
                trial_S = S[:idx] + S[idx+1:]
                if not trial_S:
                    continue
                if _is_valid_set(trial_S, dist) and _is_maximal(trial_S, dist, pair_conflicts):
                    S = trial_S
                    improved = True
                    break
        return S
    
    # Apply local search to best solution
    if best_set is not None:
        best_set = _local_search(best_set, 10.0)
        if _is_valid_set(best_set, dist) and _is_maximal(best_set, dist, pair_conflicts):
            best_size = len(best_set)
    
    # Now try to find smaller solutions via systematic search
    # Use a different approach: start from empty set, add points strategically
    
    # For now, return the best we have
    if best_set is None:
        return []
    
    return [points[i] for i in best_set]

def construct():
    points, dist, pair_conflicts = _precompute()
    return _construct_bnb(dist, points, pair_conflicts)

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
