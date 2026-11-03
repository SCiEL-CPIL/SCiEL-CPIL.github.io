import time
import numpy as np
from itertools import combinations

D = 11

def _generate_weight_m_vectors(m):
    """Generate all integer vectors in R^D with exactly m non-zero entries of value 1 or -1."""
    if m > D:
        return np.empty((0, D), dtype=np.int64)
    
    vecs = []
    indices = range(D)
    for comb in combinations(indices, m):
        for mask in range(1 << m):
            v = np.zeros(D, dtype=np.int64)
            for j, idx in enumerate(comb):
                if mask & (1 << j):
                    v[idx] = 1
                else:
                    v[idx] = -1
            vecs.append(v)
    return np.array(vecs, dtype=np.int64)

def construct():
    deadline = time.time() + 110  # Leave 10s margin for safety
    
    # Generate pool of weight 4 vectors
    # Weight 4 has norm sqrt(4) = 2.
    # Condition for valid pair u, v: ||u-v|| >= 2 => ||u-v||^2 >= 4
    # 4 + 4 - 2 u.v >= 4 => u.v <= 2
    # So we must AVOID pairs with u.v >= 3.
    # Max u.v for weight 4 is 4 (identical). Next is 3 (share 3 indices, signs align).
    # So conflict if u.v >= 3.
    
    V = _generate_weight_m_vectors(4)
    N = len(V)
    
    if N == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    # Build adjacency bitmasks
    # adj[i] is an integer with bit j set if V[i] and V[j] conflict.
    # We only need to check j > i for symmetry, but bitmask is easier if we fill all.
    # N = 5280. Integer size is ~ 5280 bits = 660 bytes.
    # 5280 * 660 bytes = ~3.5 MB. Very small.
    
    # Compute inner products
    # Dots = V @ V.T  (5280 x 5280) = 28M entries. Memory ~ 224 MB (int64) or 112 MB (int32).
    # This might be slow to compute and store.
    # Alternative: Chunked computation or sparse logic.
    # Let's try chunked to be safe on memory/time.
    
    adj = [0] * N
    
    # Conflict condition: dot >= 3
    # We can compute this in blocks of rows.
    block_size = 500
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        V_block = V[start:end]
        # Compute dots for this block against ALL vectors
        # Dots_block = V_block @ V.T
        dots = V_block @ V.T
        
        # Identify conflicts: dots >= 3
        # Exclude self (diagonal is 4, which is >= 3, but self is not a conflict with another)
        # Actually, we don't add self to adj.
        
        for i in range(end - start):
            row = dots[i]
            # We want to set bits for all j where row[j] >= 3 and j != i
            conflict_mask = (row >= 3).astype(np.uint8)
            # Clear self bit
            conflict_mask[i + start] = 0
            
            # Convert boolean array to integer bitmask
            # np.packbits is efficient
            packed = np.packbits(conflict_mask)
            # Python int from bytes
            # Need to handle endianness. Little-endian is standard for this task usually.
            adj[i + start] = int.from_bytes(packed, byteorder='little')
        
        if time.time() > deadline - 20:
            break

    # Heuristic: Greedy + Local Search
    # We maintain a set S (indices) and a forbidden mask.
    # forbidden mask has bit j set if any u in S has adj[u] with bit j set.
    
    best_size = 0
    best_set = []
    
    rng = np.random.default_rng(42)
    
    # Try multiple restarts or long single run?
    # Given N=5280, a good greedy gets maybe 1000+.
    # Local search can improve.
    
    while time.time() < deadline - 5:
        # Initialize
        S = set()
        forbidden = 0
        
        # Greedy initialization
        # Random order
        order = rng.permutation(N)
        for v in order:
            # Check if v is forbidden
            if not ((forbidden >> v) & 1):
                S.add(v)
                forbidden |= adj[v]
        
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_set = list(S)
            
        # Local Search
        # Moves:
        # 1. Try to add a random v not in S and not forbidden.
        # 2. Try to remove u in S, then try to add 2 random candidates that were blocked by u.
        
        # Precompute a list of candidates? No, just random sampling.
        
        steps = 0
        max_steps = 5000 # Adjust based on time
        
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            
            # Move 1: Add
            # Pick a random vertex not in S
            # Efficient way: just pick random indices until we find one not in S and not forbidden.
            # Since S is small compared to N, this is fast.
            improved = False
            for _ in range(10):
                v = int(rng.integers(N))
                if v in S:
                    continue
                if (forbidden >> v) & 1:
                    continue
                
                # Can add
                S.add(v)
                forbidden |= adj[v]
                current_size += 1
                improved = True
                break
            
            if improved:
                if current_size > best_size:
                    best_size = current_size
                    best_set = list(S)
                continue
            
            # Move 2: Swap (Remove 1, Add 2)
            # Pick u to remove
            u = int(rng.choice(list(S)))
            
            # Remove u
            S.remove(u)
            # Recalculate forbidden?
            # We can't just subtract adj[u] because other nodes in S might also block the same neighbors.
            # We must recompute forbidden from S.
            # O(|S|) big-int ops.
            new_forbidden = 0
            for node in S:
                new_forbidden |= adj[node]
            
            # Now try to add 2 vertices that were forbidden by u but maybe not by others?
            # Or just any vertex not in S and not in new_forbidden.
            # Ideally, vertices that were blocked by u.
            # Candidates: bits in adj[u] that are 0 in new_forbidden.
            # cand_mask = adj[u] & ~new_forbidden
            # This is efficient.
            
            cand_mask = adj[u] & ~new_forbidden
            
            # If cand_mask is 0, we didn't free up anything. Revert.
            # We need to pick 2 vertices from cand_mask.
            # Finding bits in a large int:
            # v1 = lowbit(cand_mask)
            # cand_mask &= ~v1
            # v2 = lowbit(cand_mask)
            
            if cand_mask == 0:
                # Revert
                S.add(u)
                forbidden = new_forbidden | adj[u]
                continue
                
            # Get v1
            lsb = cand_mask & -cand_mask
            v1 = lsb.bit_length() - 1
            cand_mask ^= lsb
            
            # If no more candidates, revert? Or just add 1?
            # We need net gain +1. So we must add 2.
            if cand_mask == 0:
                S.add(u)
                forbidden = new_forbidden | adj[u]
                continue
                
            lsb2 = cand_mask & -cand_mask
            v2 = lsb2.bit_length() - 1
            
            # Check if v1, v2 are in S? They shouldn't be, as they were forbidden before (implied by being in adj[u]? No, they were in forbidden. If they were in S, they wouldn't be in forbidden? No, S nodes are not forbidden from themselves, but they are in S. 
            # Actually, if v1 was in S, it wouldn't be in adj[u] unless u conflicts with v1. But if v1 in S and u in S, they don't conflict. So v1 not in adj[u].
            # So v1, v2 are not in S.
            
            # Add v1, v2
            S.add(v1)
            S.add(v2)
            
            # Update forbidden
            new_forbidden |= adj[v1]
            new_forbidden |= adj[v2]
            forbidden = new_forbidden
            
            current_size += 1 # Removed 1, Added 2 -> Net +1
            
            if current_size > best_size:
                best_size = current_size
                best_set = list(S)

    # Construct result array
    if not best_set:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    indices = np.array(best_set, dtype=np.int64)
    result = V[indices]
    
    return result

# EVOLVE-BLOCK-END
