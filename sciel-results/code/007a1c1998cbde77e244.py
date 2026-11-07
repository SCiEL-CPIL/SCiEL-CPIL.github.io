"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    n = 10
    
    # Generate candidates based on linear combinations over Z5 x Z2
    # L[i][j] = (a_i + m*a_j + n*b_j) % 5, (b_i + b_j) % 2
    # m in {1,2,3,4}, n in {0,1,2,3,4}
    all_squares = []
    for m in [1, 2, 3, 4]:
        for n_val in [0, 1, 2, 3, 4]:
            sq = [[0]*10 for _ in range(10)]
            for i in range(10):
                for j in range(10):
                    a_i = i % 5
                    b_i = i // 5
                    a_j = j % 5
                    b_j = j // 5
                    
                    val_a = (a_i + m * a_j + n_val * b_j) % 5
                    val_b = (b_i + b_j) % 2
                    sq[i][j] = val_a + 5 * val_b
            all_squares.append(sq)
    
    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(10):
            for j in range(10):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True
    
    # Backtracking search for maximum orthogonal subset
    best_set = []
    
    def backtrack(current_set, remaining_indices):
        nonlocal best_set
        if len(current_set) > len(best_set):
            best_set = list(current_set)
        if len(best_set) == 9:
            return
        if not remaining_indices:
            return
            
        for idx in remaining_indices:
            sq = all_squares[idx]
            # Check orthogonality with all in current_set
            is_orth = True
            for sel in current_set:
                if not are_orthogonal(sq, sel):
                    is_orth = False
                    break
            if is_orth:
                current_set.append(sq)
                new_remaining = [r for r in remaining_indices if r != idx]
                backtrack(current_set, new_remaining)
                current_set.pop()
                
    backtrack([], list(range(len(all_squares))))
    
    return best_set

# EVOLVE-BLOCK-END
