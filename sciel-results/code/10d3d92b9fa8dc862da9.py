import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s for safety

    best_n = 17
    best_colors = [0 if (j - i) % 17 in {x * x % 17 for x in range(1, 17)} else 1 for i in range(17) for j in range(i + 1, 17)]

    # Try to find a valid coloring for n=30, then 29, etc.
    for target_n in [30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18]:
        if time.time() > deadline - 15:
            break
            
        n = target_n
        num_edges = n * (n - 1) // 2
        
        # Precompute index matrix
        idx_mat = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx_mat[i][j] = idx
                idx_mat[j][i] = idx
                idx += 1

        found = False
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize random coloring
            colors = [random.randint(0, 1) for _ in range(num_edges)]
            
            # Local search
            for step in range(20000):
                if time.time() > deadline - 5:
                    break
                    
                # Check validity and find a conflict
                valid = True
                conflict_found = False
                
                for vertices in itertools.combinations(range(n), 6):
                    # Check all 15 partitions
                    for p1 in range(3):
                        a1 = vertices[p1]
                        a2 = vertices[p1+1]
                        remaining = [v for k, v in enumerate(vertices) if k != p1 and k != p1+1]
                        for p2 in range(2):
                            b1 = remaining[p2]
                            b2 = remaining[p2+1]
                            c1 = remaining[3-p2]
                            c2 = remaining[4-p2]
                            
                            # 12 edges
                            e_ab1 = colors[idx_mat[a1][b1]]
                            e_ab2 = colors[idx_mat[a1][b2]]
                            e_a2b1 = colors[idx_mat[a2][b1]]
                            e_a2b2 = colors[idx_mat[a2][b2]]
                            e_ac1 = colors[idx_mat[a1][c1]]
                            e_ac2 = colors[idx_mat[a1][c2]]
                            e_a2c1 = colors[idx_mat[a2][c1]]
                            e_a2c2 = colors[idx_mat[a2][c2]]
                            e_bc1 = colors[idx_mat[b1][c1]]
                            e_bc2 = colors[idx_mat[b1][c2]]
                            e_b2c1 = colors[idx_mat[b2][c1]]
                            e_b2c2 = colors[idx_mat[b2][c2]]
                            
                            edges = [e_ab1, e_ab2, e_a2b1, e_a2b2, e_ac1, e_ac2, e_a2c1, e_a2c2, e_bc1, e_bc2, e_b2c1, e_b2c2]
                            if all(e == edges[0] for e in edges):
                                valid = False
                                conflict_found = True
                                # Flip a random edge in this conflict
                                pairs = [
                                    (a1, b1), (a1, b2), (a2, b1), (a2, b2),
                                    (a1, c1), (a1, c2), (a2, c1), (a2, c2),
                                    (b1, c1), (b1, c2), (b2, c1), (b2, c2)
                                ]
                                u, v = random.choice(pairs)
                                idx = idx_mat[u][v]
                                colors[idx] = 1 - colors[idx]
                                break
                        if conflict_found:
                            break
                    if conflict_found:
                        break
                    if not valid:
                        break
                
                if valid:
                    found = True
                    best_n = n
                    best_colors = colors[:]
                    break
                if time.time() > deadline - 5:
                    break

            if found:
                break

    return {'n': best_n, 'colors': best_colors}
