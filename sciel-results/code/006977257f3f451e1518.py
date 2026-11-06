"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

    n = 16
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: i for i, e in enumerate(all_pairs)}

    # ---- Precompute Q3 masks (12-edge 3-regular submasks of K4,4) ----
    def is_q3_mask(mask):
        da = [0]*4
        db = [0]*4
        for i in range(4):
            for j in range(4):
                if mask & (1 << (i*4 + j)):
                    da[i] += 1
                    db[j] += 1
        return all(d == 3 for d in da) and all(d == 3 for d in db)

    q3_masks = []
    for combo in itertools.combinations(range(16), 12):
        m = 0
        for idx in combo:
            m |= (1 << idx)
        if is_q3_mask(m):
            q3_masks.append(m)
    q3_masks = tuple(q3_masks)  # 6 masks

    # ---- Precompute all K4,4 splits (A of size 4, B of size 4) ----
    all_verts = list(range(n))
    k44_list = []  # each: (A tuple, B tuple, pairs [(a,b,bitpos),...])
    for A in itertools.combinations(all_verts, 4):
        Aset = set(A)
        for B in itertools.combinations(all_verts, 4):
            if Aset & set(B):
                continue
            pairs = []
            for i in range(4):
                for j in range(4):
                    pairs.append((A[i], B[j], i*4 + j))
            k44_list.append((A, B, pairs))

    # ---- Index: for each edge, which K4,4 splits contain it ----
    edge_to_k44 = {}
    for idx, (A, B, pairs) in enumerate(k44_list):
        for a, b, _ in pairs:
            e = (a, b) if a < b else (b, a)
            edge_to_k44.setdefault(e, []).append(idx)

    # ---- Build graph as bitsets ----
    adj = [0] * n
    def has_edge(u, v):
        return bool(adj[u] & (1 << v))
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    def edge_count():
        return sum(bin(adj[i]).count("1") for i in range(n)) // 2

    def time_left():
        return deadline - time.time()

    # ---- Check if adding (u,v) creates a Q3 ----
    def creates_q3(u, v):
        e = (u, v) if u < v else (v, u)
        for kidx in edge_to_k44.get(e, ()):
            A, B, pairs = k44_list[kidx]
            mask = 0
            for a, b, bit in pairs:
                if (a == u and b == v) or (a == v and b == u):
                    mask |= (1 << bit)  # the new edge
                elif has_edge(a, b):
                    mask |= (1 << bit)
            for qm in q3_masks:
                if (mask & qm) == qm:
                    return True
        return False

    # ---- Count number of Q3 subgraphs in current graph ----
    def count_q3():
        cnt = 0
        for kidx in range(len(k44_list)):
            A, B, pairs = k44_list[kidx]
            mask = 0
            for a, b, bit in pairs:
                if has_edge(a, b):
                    mask |= (1 << bit)
            for qm in q3_masks:
                if (mask & qm) == qm:
                    cnt += 1
        return cnt

    # ---- Build Clebsch complement as initial graph ----
    # Clebsch graph: vertices are 4-bit strings (0..15), adjacent if Hamming distance = 1 or 3
    # Actually, the Clebsch graph has vertices as binary strings of length 4,
    # and two vertices are adjacent if they differ in exactly 1 or 3 positions.
    # Wait, let me use the standard definition:
    # The Clebsch graph is the folded 5-cube. Vertices are 4-bit strings.
    # Two vertices are adjacent if their Hamming distance is 1 or 3.
    # This gives a 5-regular graph (4 neighbors at distance 1, 1 neighbor at distance 3... 
    # Actually: for a 4-bit string, there are 4 strings at distance 1 and C(4,3)=4 strings at distance 3.
    # So degree = 4 + 4 = 8? That's not right.
    # Let me reconsider. The Clebsch graph has parameters (16,5,0,2).
    # Standard construction: vertices are {0,1}^4. Two vertices are adjacent if they differ
    # in exactly 1 coordinate, OR if they are complementary (differ in all 4 coordinates).
    # Degree: 4 (flip one bit) + 1 (flip all 4) = 5. Yes!
    
    # So Clebsch graph edges: (u,v) where hamming_distance(u,v) == 1 or hamming_distance(u,v) == 4
    # Complement: (u,v) where hamming_distance(u,v) == 2 or hamming_distance(u,v) == 3
    
    def hamming_dist(a, b):
        return bin(a ^ b).count('1')
    
    # Build complement of Clebsch graph
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            d = hamming_dist(i, j)
            if d == 2 or d == 3:
                add_edge(i, j)
    
    initial_edges = edge_count()
    
    # Greedily remove edges to make Q3-free
    # Find all edges and try removing ones that are in Q3s
    all_current_edges = [(i, j) for i in range(n) for j in range(i+1, n) if has_edge(i, j)]
    random.seed(42)
    random.shuffle(all_current_edges)
    
    # First pass: remove edges that participate in Q3s
    for u, v in all_current_edges:
        if time_left() < 30:
            break
        if count_q3() == 0:
            break
        remove_edge(u, v)
    
    # Now try to add back edges we removed
    for u, v in reversed(all_current_edges):
        if time_left() < 25:
            break
        if not has_edge(u, v):
            if not creates_q3(u, v):
                add_edge(u, v)
    
    best_edges = edge_count()
    best_adj = adj[:]
    
    # Also try the K7+K7+K2 seed as a fallback
    adj2 = [0] * n
    for i in range(7):
        for j in range(i+1, 7):
            adj2[i] |= (1 << j)
            adj2[j] |= (1 << i)
    for i in range(7, 14):
        for j in range(i+1, 14):
            adj2[i] |= (1 << j)
            adj2[j] |= (1 << i)
    adj2[14] |= (1 << 15)
    adj2[15] |= (1 << 14)
    
    seed_edges = sum(bin(adj2[i]).count("1") for i in range(n)) // 2
    if seed_edges > best_edges:
        best_edges = seed_edges
        best_adj = adj2[:]
    
    # ---- Simulated Annealing ----
    if time_left() > 20:
        adj[:] = best_adj[:]
        sa_edges = edge_count()
        sa_q3 = count_q3()  # should be 0
        
        # Temperature schedule - start higher to explore more
        T = 8.0
        cooling = 0.9997
        min_T = 0.05
        
        all_pairs_list = all_pairs[:]
        
        while time_left() > 10 and T > min_T:
            # Pick a random edge to toggle
            u, v = random.choice(all_pairs_list)
            if has_edge(u, v):
                # Try removing
                remove_edge(u, v)
                new_edges = sa_edges - 1
                new_q3 = count_q3()
                new_score = new_edges - 50 * new_q3
                old_score = sa_edges - 50 * sa_q3
                delta = new_score - old_score
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    sa_edges = new_edges
                    sa_q3 = new_q3
                    if sa_q3 == 0 and sa_edges > best_edges:
                        best_edges = sa_edges
                        best_adj = adj[:]
                else:
                    add_edge(u, v)
            else:
                # Try adding
                add_edge(u, v)
                new_edges = sa_edges + 1
                new_q3 = count_q3()
                new_score = new_edges - 50 * new_q3
                old_score = sa_edges - 50 * sa_q3
                delta = new_score - old_score
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    sa_edges = new_edges
                    sa_q3 = new_q3
                    if sa_q3 == 0 and sa_edges > best_edges:
                        best_edges = sa_edges
                        best_adj = adj[:]
                else:
                    remove_edge(u, v)
            
            T *= cooling
        
        # After SA, ensure we have a valid Q3-free graph
        if count_q3() > 0:
            for _ in range(200):
                if count_q3() == 0:
                    break
                if time_left() < 5:
                    break
                found = False
                for u, v in all_pairs:
                    if not has_edge(u, v):
                        continue
                    old_q3 = count_q3()
                    remove_edge(u, v)
                    new_q3 = count_q3()
                    if new_q3 < old_q3:
                        found = True
                        break
                    add_edge(u, v)
                if not found:
                    break
    
    # ---- Local search: 1-1 edge swaps ----
    adj[:] = best_adj[:]
    for _ in range(3000):
        if time_left() < 10:
            break
        rem_edges = [e for e in all_pairs if has_edge(*e)]
        if not rem_edges:
            break
        ru, rv = random.choice(rem_edges)
        remove_edge(ru, rv)
        
        miss = [e for e in all_pairs if not has_edge(*e)]
        random.shuffle(miss)
        added = 0
        for u, v in miss:
            if added >= 2:
                break
            if not creates_q3(u, v):
                add_edge(u, v)
                added += 1
        
        if edge_count() > best_edges:
            best_edges = edge_count()
            best_adj = adj[:]
        else:
            for u, v in miss[:added]:
                remove_edge(u, v)
            add_edge(ru, rv)
    
    # ---- Second SA pass with lower temperature for fine-tuning ----
    if time_left() > 15:
        adj[:] = best_adj[:]
        sa_edges = edge_count()
        sa_q3 = count_q3()
        
        T = 2.0
        cooling = 0.9995
        min_T = 0.01
        
        while time_left() > 8 and T > min_T:
            u, v = random.choice(all_pairs_list)
            if has_edge(u, v):
                remove_edge(u, v)
                new_edges = sa_edges - 1
                new_q3 = count_q3()
                new_score = new_edges - 50 * new_q3
                old_score = sa_edges - 50 * sa_q3
                delta = new_score - old_score
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    sa_edges = new_edges
                    sa_q3 = new_q3
                    if sa_q3 == 0 and sa_edges > best_edges:
                        best_edges = sa_edges
                        best_adj = adj[:]
                else:
                    add_edge(u, v)
            else:
                add_edge(u, v)
                new_edges = sa_edges + 1
                new_q3 = count_q3()
                new_score = new_edges - 50 * new_q3
                old_score = sa_edges - 50 * sa_q3
                delta = new_score - old_score
                if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                    sa_edges = new_edges
                    sa_q3 = new_q3
                    if sa_q3 == 0 and sa_edges > best_edges:
                        best_edges = sa_edges
                        best_adj = adj[:]
                else:
                    remove_edge(u, v)
            
            T *= cooling
        
        if count_q3() > 0:
            for _ in range(100):
                if count_q3() == 0:
                    break
                if time_left() < 5:
                    break
                found = False
                for u, v in all_pairs:
                    if not has_edge(u, v):
                        continue
                    old_q3 = count_q3()
                    remove_edge(u, v)
                    new_q3 = count_q3()
                    if new_q3 < old_q3:
                        found = True
                        break
                    add_edge(u, v)
                if not found:
                    break
    
    # Restore best state
    adj[:] = best_adj[:]
    
    # Final validity check
    if time_left() > 5:
        found = False
        for kidx in range(len(k44_list)):
            A, B, pairs = k44_list[kidx]
            mask = 0
            for a, b, bit in pairs:
                if has_edge(a, b):
                    mask |= (1 << bit)
            for qm in q3_masks:
                if (mask & qm) == qm:
                    found = True
                    break
            if found:
                break
        if found:
            # Fallback to seed
            adj = [0]*n
            for i in range(7):
                for j in range(i+1, 7):
                    add_edge(i, j)
            for i in range(7, 14):
                for j in range(i+1, 14):
                    add_edge(i, j)
            add_edge(14, 15)
    
    return [[i, j] for i in range(n) for j in range(i+1, n) if has_edge(i, j)]
# EVOLVE-BLOCK-END
