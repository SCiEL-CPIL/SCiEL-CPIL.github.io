# EVOLVE-BLOCK-START
import random

def construct():
    n = 10
    
    # Generate candidate squares
    # Form: L(i, j) = ((a_i + m*a_j + n*b_j) % 5, (b_i + p*b_j) % 2)
    # m in 1..4, n in 0..4, p in 0..1
    candidates = []
    for m in range(1, 5):
        for n_val in range(5):
            for p in range(2):
                sq = [[0]*10 for _ in range(10)]
                for i in range(10):
                    a_i = i % 5
                    b_i = i // 5
                    for j in range(10):
                        a_j = j % 5
                        b_j = j // 5
                        val_a = (a_i + m * a_j + n_val * b_j) % 5
                        val_b = (b_i + p * b_j) % 2
                        sq[i][j] = val_a + 5 * val_b
                candidates.append(sq)
                
    # Check orthogonality
    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(10):
            for j in range(10):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return len(pairs) == 100

    # Build orthogonality graph
    num_cands = len(candidates)
    ortho = [[False]*num_cands for _ in range(num_cands)]
    for i in range(num_cands):
        for j in range(i+1, num_cands):
            if are_orthogonal(candidates[i], candidates[j]):
                ortho[i][j] = True
                ortho[j][i] = True
                
    # Find maximum clique using backtracking
    # Since num_cands is small (40), we can use a simple recursive search
    # Or greedy with restarts.
    # Let's try a simple backtracking for max clique.
    
    best_clique = []
    
    def backtrack(current_clique, available):
        nonlocal best_clique
        if len(current_clique) > len(best_clique):
            best_clique = current_clique[:]
        if not available:
            return
            
        # Optimization: if len(current_clique) + len(available) <= len(best_clique), return
        if len(current_clique) + len(available) <= len(best_clique):
            return
            
        # Pick a vertex from available
        # To find max clique, we should pick vertices carefully.
        # Simple approach: try each vertex
        for idx in range(len(available)):
            v = available[idx]
            new_available = [u for u in available[idx+1:] if ortho[v][u]]
            backtrack(current_clique + [v], new_available)
            
    # Initial available set is all indices
    # To speed up, we can sort or use a heuristic. 
    # Given 40 nodes, worst case is high, but graph is sparse? 
    # Actually, orthogonality is a strong condition.
    # Let's use a greedy approach first, then local search?
    # Or just simple backtracking with pruning.
    
    # To avoid timeout, we limit the search or use a better algorithm.
    # A simple greedy might get stuck at 4 or 5.
    # Let's use a randomized greedy multiple times.
    
    best_size = 0
    best_set = []
    
    for _ in range(1000): # 1000 restarts
        available = list(range(num_cands))
        random.shuffle(available)
        current = []
        for v in available:
            if all(ortho[v][u] for u in current):
                current.append(v)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            if best_size == 9: # Max possible
                break
                
    return [candidates[i] for i in best_set]

# EVOLVE-BLOCK-END
