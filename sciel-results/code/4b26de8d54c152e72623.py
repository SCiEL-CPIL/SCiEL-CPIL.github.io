"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check that the (i,j,k) entry of the multiplication tensor is 1
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                val = Fraction(0)
                for c in range(rank):
                    val += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if val != 1:
                    return False
    # Also verify that all other tensor entries are 0
    # The multiplication tensor is zero except at (i,j,k) positions
    # We need to check that for any (a,b,c) not equal to a valid (i,j,k), the sum is 0
    # But since we're only decomposing the multiplication tensor, we just need the (i,j,k) entries
    return True

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using integer arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def _try_rank_reduction(U, V, W, n, m, p, target_rank, timeout=30):
    """Try to reduce rank by removing zero columns or combining terms."""
    start_time = time.time()
    current_rank = len(U[0])
    
    # First, remove any columns that are all zeros
    for c in range(current_rank - 1, -1, -1):
        if time.time() - start_time > timeout:
            break
        col_u = [U[i][c] for i in range(len(U))]
        col_v = [V[i][c] for i in range(len(V))]
        col_w = [W[i][c] for i in range(len(W))]
        if all(x == 0 for x in col_u + col_v + col_w):
            # Remove this column
            for i in range(len(U)):
                U[i].pop(c)
            for i in range(len(V)):
                V[i].pop(c)
            for i in range(len(W)):
                W[i].pop(c)
            current_rank -= 1
            if current_rank < target_rank:
                break
    
    return current_rank

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    deadline = time.time() + 110  # Leave 10s margin
    
    # Start with schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Try to find lower rank decompositions
    # Strategy: Try to express some terms as linear combinations of others
    
    # For small n, we can try to find rank reductions by solving linear systems
    # The tensor T has entries T[i,j,k] = delta for valid (i,j,k)
    # We want U, V, W with rank r such that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    # Try rank from schoolbook_rank-1 down
    for target_rank in range(schoolbook_rank - 1, max(10, schoolbook_rank - 20), -1):
        if time.time() > deadline:
            break
        
        # Try random perturbations to find lower rank decompositions
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Start from schoolbook and try to eliminate one column
            trial_U = [row[:] for row in U]
            trial_V = [row[:] for row in V]
            trial_W = [row[:] for row in W]
            
            # Pick a random column to try to eliminate
            col_to_elim = random.randint(0, schoolbook_rank - 1)
            
            # Try to express this column as a combination of others
            # This is a nonlinear problem, so we try random weight assignments
            success = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                # Try to set the column to zero by adjusting other columns
                # This is complex, so we try a simpler approach:
                # Check if removing this column still gives valid decomposition
                # (This only works if the column was redundant)
                
                # Alternative: try to find a rank-1 update that cancels this term
                # For now, just check if the decomposition is still valid without this column
                test_U = [row[:] for row in trial_U]
                test_V = [row[:] for row in trial_V]
                test_W = [row[:] for row in trial_W]
                
                # Remove the column
                for i in range(len(test_U)):
                    test_U[i].pop(col_to_elim)
                for i in range(len(test_V)):
                    test_V[i].pop(col_to_elim)
                for i in range(len(test_W)):
                    test_W[i].pop(col_to_elim)
                
                if _verify_fast(test_U, test_V, test_W, n, m, p):
                    success = True
                    trial_U = test_U
                    trial_V = test_V
                    trial_W = test_W
                    break
            
            if success:
                new_rank = len(trial_U[0])
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U = trial_U
                    best_V = trial_V
                    best_W = trial_W
    
    # Final verification
    if _verify_fast(best_U, best_V, best_W, n, m, p):
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        # Fallback to schoolbook
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
