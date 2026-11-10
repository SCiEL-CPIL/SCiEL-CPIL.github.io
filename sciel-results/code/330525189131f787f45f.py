import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # For k=5, g=9, we attempt to find a cyclic Cayley graph.
    # If not found, fallback to LUW with corrected d.
    
    if k == 5 and g == 9:
        # Try to find a cyclic Cayley graph with n < 1250
        # Known best is 1116.
        
        best_n = 1250 # Default to LUW size
        best_edges = None
        
        # We will try to find a valid cyclic graph
        # Iterate n from 1116 to 1249
        # Step 2 to keep n even
        
        # To save time, we try a subset or limited trials.
        # Given the difficulty, we might just skip this if it's too slow.
        # But let's try a few.
        
        # Try to find a valid cyclic graph
        # Use a more efficient search strategy
        
        # First, try some known good constructions or patterns
        # For n=1116, try specific generator sets that are known to work or likely work
        
        # Strategy: for each n, try a few carefully chosen generator sets
        # The key insight is that for a 5-regular Cayley graph on Z_n,
        # S must contain n/2 (the "diameter" element) and two other pairs {a, -a}, {b, -b}
        
        # To maximize girth, we want a, b, n/2 to be "spread out"
        # A good heuristic: a ≈ n/3, b ≈ n/5 or similar
        
        for n in range(1116, 1250, 2):
            if time.time() > deadline - 15:  # Leave 15s for fallback
                break
                
            # Try several structured choices first
            candidates = []
            
            # Pattern 1: a = n//3, b = n//5 (approximate)
            a1 = n // 3
            b1 = n // 5
            if a1 != b1 and a1 != n//2 and b1 != n//2:
                candidates.append((a1, b1))
            
            # Pattern 2: a = n//3 + 1, b = n//5 + 1
            a2 = n // 3 + 1
            b2 = n // 5 + 1
            if a2 != b2 and a2 != n//2 and b2 != n//2:
                candidates.append((a2, b2))
            
            # Pattern 3: a = n//4, b = 3*n//7
            a3 = n // 4
            b3 = 3 * n // 7
            if a3 != b3 and a3 != n//2 and b3 != n//2:
                candidates.append((a3, b3))
            
            # Add some random candidates
            for _ in range(3):
                if time.time() > deadline - 15:
                    break
                low = n // 4
                high = 3 * n // 4
                valid_range = list(range(low, high))
                if n // 2 in valid_range:
                    valid_range.remove(n // 2)
                if len(valid_range) >= 2:
                    s1, s2 = random.sample(valid_range, 2)
                    if s2 != (-s1) % n:
                        candidates.append((s1, s2))
            
            for s1, s2 in candidates:
                if time.time() > deadline - 15:
                    break
                    
                S = [n // 2, s1, (-s1) % n, s2, (-s2) % n]
                if len(set(S)) != 5:
                    continue
                
                # Check girth
                if _check_girth_numpy(n, S, g):
                    edges = []
                    for u in range(n):
                        for s in S:
                            v = (u + s) % n
                            if u < v:
                                edges.append((u, v))
                    return n, edges
                    
        # If no cyclic graph found, return LUW with corrected d
        return _luw_construct(k, g)
        
    # Fallback for other k, g
    return _luw_construct(k, g)

def _check_girth_numpy(n, S, g):
    """Check if the Cayley graph on Z_n with generating set S has girth >= g.
    
    Uses BFS-based girth detection which is much faster than enumerating all walks.
    """
    S_set = set(S)
    S_arr = list(S)
    
    # For each starting vertex, do BFS to find the shortest cycle
    # A cycle of length L through vertex 0 corresponds to a non-backtracking walk
    # of length L that returns to 0, where the first step is not the reverse of the last.
    
    # More efficient: check all walks of length L from 0 back to 0, ensuring
    # no backtracking and no shorter return.
    
    for L in range(3, g):
        # Enumerate all sequences of length L from S
        # But do it incrementally with early termination
        
        # Use iterative deepening with pruning
        # State: (current_position, last_step, visited_positions_set)
        # But this is still exponential...
        
        # Better approach: for small L (3 to 8), enumerate all S^L sequences
        # but check the sum and partial sums efficiently
        
        # For L=3: s1+s2+s3=0 mod n, with s1,s2,s3 in S
        # Check no backtracking: s2 != -s1, s3 != -s2
        # Check no shorter cycle: s1 != 0 (always true since S has no 0), 
        #   s1+s2 != 0 (else cycle of length 2, but graph is simple so this can't happen if -s1 not in S)
        
        # Actually for a simple graph with S symmetric (s in S implies -s in S),
        # a 2-cycle would mean s and -s both in S, giving edge u->u+s and u+s->u which is the same edge.
        # So no 2-cycles.
        # A 3-cycle: s1+s2+s3=0, s1,s2,s3 in S, with s1!=s2, s2!=s3, s3!=s1 (no backtracking means s2 != -s1, s3 != -s2)
        
        # Let's just enumerate but with numpy vectorization per L
        
        # Generate all sequences of length L
        # For L up to 8, 5^8 = 390625 which is manageable
        
        # Incremental approach: build up sequences
        if L == 3:
            # All triples (s1, s2, s3) in S^3
            s1 = np.array(S_arr)
            s2 = np.array(S_arr)
            s3 = np.array(S_arr)
            
            # Expand to all combinations
            s1_exp = np.repeat(s1, len(S_arr) * len(S_arr))
            s2_exp = np.tile(np.repeat(s2, len(S_arr)), len(S_arr))
            s3_exp = np.tile(s3, len(S_arr) * len(S_arr))
            
            total = (s1_exp + s2_exp + s3_exp) % n
            mask = total == 0
            
            if not mask.any():
                continue
                
            s1_m = s1_exp[mask]
            s2_m = s2_exp[mask]
            s3_m = s3_exp[mask]
            
            # Check no backtracking: s2 != -s1 mod n, s3 != -s2 mod n
            bt1 = (s2_m + s1_m) % n == 0
            bt2 = (s3_m + s2_m) % n == 0
            valid = ~(bt1 | bt2)
            
            if valid.any():
                # Check no shorter cycle: partial sums s1, s1+s2 should not be 0
                ps1 = s1_m[valid]
                ps2 = (ps1 + s2_m[valid]) % n
                if (ps1 == 0).any() or (ps2 == 0).any():
                    # This would mean a shorter cycle exists
                    # But ps1=0 means s1=0 which is impossible
                    # ps2=0 means s1+s2=0, i.e., s2=-s1, which is backtracking
                    # So if backtracking is already excluded, this shouldn't happen
                    pass
                # All checks passed, we have a cycle of length L
                return False
        
        elif L == 4:
            s1 = np.array(S_arr)
            s2 = np.array(S_arr)
            s3 = np.array(S_arr)
            s4 = np.array(S_arr)
            
            s1_exp = np.repeat(s1, len(S_arr)**3)
            s2_exp = np.tile(np.repeat(s2, len(S_arr)**2), len(S_arr))
            s3_exp = np.tile(np.repeat(s3, len(S_arr)), len(S_arr)**2)
            s4_exp = np.tile(s4, len(S_arr)**3)
            
            total = (s1_exp + s2_exp + s3_exp + s4_exp) % n
            mask = total == 0
            
            if not mask.any():
                continue
                
            s1_m = s1_exp[mask]
            s2_m = s2_exp[mask]
            s3_m = s3_exp[mask]
            s4_m = s4_exp[mask]
            
            bt1 = (s2_m + s1_m) % n == 0
            bt2 = (s3_m + s2_m) % n == 0
            bt3 = (s4_m + s3_m) % n == 0
            valid = ~(bt1 | bt2 | bt3)
            
            if valid.any():
                ps1 = s1_m[valid]
                ps2 = (ps1 + s2_m[valid]) % n
                ps3 = (ps2 + s3_m[valid]) % n
                if (ps1 == 0).any() or (ps2 == 0).any() or (ps3 == 0).any():
                    pass
                return False
        
        else:
            # For L >= 5, use a different approach: 
            # Count walks of length L from 0 to 0 using matrix powers
            # but subtract backtracking walks
            
            # Actually, let's use a smarter enumeration for larger L
            # Build sequences incrementally
            
            # Start with all sequences of length L-1 that sum to some value
            # and extend by one more step
            
            # For efficiency, use a dictionary: sum_value -> list of (last_step, partial_sums_info)
            # But this could blow up...
            
            # Alternative: for L >= 5, just sample a subset of sequences
            # Or use the fact that if there's a cycle of length L, 
            # we can detect it via the number of closed walks
            
            # Let's use matrix exponentiation approach
            # A = adjacency matrix, trace(A^L) counts closed walks of length L
            # But we need non-backtracking closed walks
            
            # For now, fall back to enumeration but with early exit
            # Generate sequences incrementally
            
            seqs = [()]
            for _ in range(L):
                new_seqs = []
                for seq in seqs:
                    for s in S_arr:
                        new_seqs.append(seq + (s,))
                seqs = new_seqs
                
            if not seqs:
                continue
                
            A = np.array(seqs, dtype=np.int64)
            sums = A.sum(axis=1) % n
            zero_indices = np.where(sums == 0)[0]
            
            if len(zero_indices) == 0:
                continue
            
            A_zero = A[zero_indices]
            partial_sums = np.cumsum(A_zero, axis=1) % n
            
            for i in range(len(zero_indices)):
                ps = partial_sums[i]
                # Check no backtracking: ps[i] != 0 for i < L-1
                # and consecutive steps don't cancel: ps[j] - ps[j-1] != -(ps[j-1] - ps[j-2]) for j>=2
                # Simpler: check that no two consecutive steps sum to 0
                steps = np.diff(np.concatenate([[0], ps]))
                for j in range(1, L):
                    if (steps[j] + steps[j-1]) % n == 0:
                        break
                else:
                    # No backtracking detected
                    # Check no shorter return: ps[i] != 0 for i < L
                    if not (ps[:-1] == 0).any():
                        return False
                        
    return True

def _luw_construct(k, g):
    q = k
    # Corrected d calculation for LUW graphs
    # Girth of G(q,d) is 2d+1
    # We need 2d+1 >= g => d >= (g-1)/2
    d = (g + 1) // 2
    if d < 3: d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
