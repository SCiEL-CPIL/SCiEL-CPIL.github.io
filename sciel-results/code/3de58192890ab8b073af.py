import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def edges_from_mask(n, mask):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = j - i
                if diff > n // 2:
                    diff = n - diff
                if (mask >> (diff - 1)) & 1:
                    edges.append([i, j])
        return edges

    def find_valid_circulant(n):
        half = n // 2
        full_mask = (1 << half) - 1
        total_masks = 1 << half
        
        # Precompute bad masks for K4
        bad_k4 = [False] * total_masks
        # Precompute bad masks for K6 (in complement, so we check K6 in mask itself)
        bad_k6 = [False] * total_masks
        
        # Generate all possible step sets as bitmasks
        # We need to check which step sets contain a K4 or K6
        
        # K4 Check:
        # A K4 in circulant graph with steps S exists if there are a,b,c in S
        # such that b-a, c-a, c-b are also in S.
        # We iterate all triples (a,b,c) with a<b<c from 1..half
        # and check if they form a K4 configuration.
        
        steps_range = range(1, half + 1)
        
        # To speed up marking, we can just iterate all masks and check?
        # No, the "mark supersets" approach is better if valid configs are sparse.
        # But checking if a triple is a valid K4 config is O(1).
        
        # Let's just iterate all masks and check validity directly, but optimized.
        # Actually, iterating 131k masks and doing a fast check is fine.
        # How to fast check K4 in a mask?
        # Extract steps? No.
        # Use bit operations.
        
        # Let's stick to the "mark bad masks" approach.
        
        # 1. Mark bad_k4
        for a in steps_range:
            ma = 1 << (a - 1)
            for b in range(a + 1, half + 1):
                mb = 1 << (b - 1)
                if b - a > half: continue # Diff must be in S, so <= half
                diff_ab = b - a
                m_diff_ab = 1 << (diff_ab - 1)
                # Check if diff_ab is a valid step (it is, if <= half)
                for c in range(b + 1, half + 1):
                    mc = 1 << (c - 1)
                    diff_ac = c - a
                    diff_bc = c - b
                    if diff_ac > half or diff_bc > half:
                        continue
                    m_diff_ac = 1 << (diff_ac - 1)
                    m_diff_bc = 1 << (diff_bc - 1)
                    
                    # The K4 requires steps a, b, c, diff_ab, diff_ac, diff_bc
                    # All these must be in the mask.
                    required_mask = ma | mb | mc | m_diff_ab | m_diff_ac | m_diff_bc
                    
                    # Mark all supersets of required_mask as bad_k4
                    # Iterate submasks of the complement?
                    # Just iterate all masks containing required_mask.
                    # free_bits = full_mask ^ required_mask
                    # for sub in range(1 << (half - bit_count(required_mask))):
                    #    ...
                    
                    # Simpler: just set bad_k4[m] = True for all m s.t. (m & required_mask) == required_mask
                    # This is 2^(17 - popcount(required_mask)) iterations.
                    # popcount is at least 3, often 6.
                    # 2^11 = 2048.
                    # Number of triples is 680.
                    # 680 * 2048 = 1.4M ops. Very fast.
                    
                    free_mask = full_mask ^ required_mask
                    # Iterate all submasks of free_mask
                    sub = free_mask
                    while True:
                        m = required_mask | sub
                        bad_k4[m] = True
                        if sub == 0: break
                        sub = (sub - 1) & free_mask
                        
        # 2. Mark bad_k6
        # A K6 in circulant graph with steps S exists if there are a,b,c,d,e in S
        # such that all pairwise diffs are in S.
        
        for a in steps_range:
            ma = 1 << (a - 1)
            for b in range(a + 1, half + 1):
                mb = 1 << (b - 1)
                diff_ab = b - a
                if diff_ab > half: continue
                m_diff_ab = 1 << (diff_ab - 1)
                for c in range(b + 1, half + 1):
                    mc = 1 << (c - 1)
                    diff_ac = c - a
                    diff_bc = c - b
                    if diff_ac > half or diff_bc > half: continue
                    m_diff_ac = 1 << (diff_ac - 1)
                    m_diff_bc = 1 << (diff_bc - 1)
                    for d in range(c + 1, half + 1):
                        md = 1 << (d - 1)
                        diff_ad = d - a
                        diff_bd = d - b
                        diff_cd = d - c
                        if diff_ad > half or diff_bd > half or diff_cd > half: continue
                        m_diff_ad = 1 << (diff_ad - 1)
                        m_diff_bd = 1 << (diff_bd - 1)
                        m_diff_cd = 1 << (diff_cd - 1)
                        for e in range(d + 1, half + 1):
                            me = 1 << (e - 1)
                            diff_ae = e - a
                            diff_be = e - b
                            diff_ce = e - c
                            diff_de = e - d
                            if diff_ae > half or diff_be > half or diff_ce > half or diff_de > half: continue
                            m_diff_ae = 1 << (diff_ae - 1)
                            m_diff_be = 1 << (diff_be - 1)
                            m_diff_ce = 1 << (diff_ce - 1)
                            m_diff_de = 1 << (diff_de - 1)
                            
                            required_mask = ma | mb | mc | md | me | m_diff_ab | m_diff_ac | m_diff_ad | m_diff_ae | m_diff_bc | m_diff_bd | m_diff_be | m_diff_cd | m_diff_ce | m_diff_de
                            
                            free_mask = full_mask ^ required_mask
                            sub = free_mask
                            while True:
                                m = required_mask | sub
                                bad_k6[m] = True
                                if sub == 0: break
                                sub = (sub - 1) & free_mask

        # Find valid mask
        for m in range(total_masks):
            if not bad_k4[m] and not bad_k6[full_mask ^ m]:
                return m
        return None

    best_n = 17
    best_edges = []
    
    # Fallback: Paley 17
    sq = {x * x % 17 for x in range(1, 17)}
    adj = [[False]*17 for _ in range(17)]
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in sq:
                adj[i][j] = True
                adj[j][i] = True
    best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if adj[i][j]]

    # Try n from 35 down to 18
    for n in range(35, 17, -1):
        if time.time() > deadline - 10:
            break
        
        mask = find_valid_circulant(n)
        if mask is not None:
            best_n = n
            best_edges = edges_from_mask(n, mask)
            break

    return {'n': best_n, 'edges': best_edges}
