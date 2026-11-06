import random
import time

def construct():
    # Construct a graph that is a disjoint union of cliques.
    # A disjoint union of cliques is C5-free if all cliques have size <= 4?
    # No, K5 has cycles of length 3, 4. Max cycle length in K_m is m-1.
    # So K5 has max cycle 4. It is C5-free.
    # K4 has max cycle 3. It is C5-free.
    # Any clique K_m is C5-free for m <= 5.
    # Independence number of K_m is 1.
    # We want alpha(G) < 9.
    # If G is a disjoint union of cliques C_1, ..., C_k, then alpha(G) = sum(alpha(C_i)) = k (since each is a clique).
    # So we need k < 9, i.e., k <= 8.
    # To maximize n, we should make the cliques as large as possible.
    # Max clique size without C5 is 5 (K5 is C5-free).
    # Can we use K6? K6 has a 5-cycle. So max clique size is 5.
    # So we want 8 cliques of size 5.
    # n = 8 * 5 = 40.
    # But the limit is n <= 35.
    # So we can use 7 cliques of size 5 and 1 clique of size 5?
    # 7 * 5 = 35.
    # Let's use 7 K5s.
    # n = 35.
    # alpha = 7 * 1 = 7 < 9.
    # C5-free? Yes, K5 has no 5-cycle (max cycle 4).
    
    n = 35
    edges = []
    # 7 disjoint K5s
    for b in range(7):
        base = b * 5
        for i in range(5):
            for j in range(i + 1, 5):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}
