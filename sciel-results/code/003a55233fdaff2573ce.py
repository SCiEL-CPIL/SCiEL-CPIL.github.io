"""Matrix multiplication 2x3x9: rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication."""
    start_time = time.time()
    DEADLINE = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    # Return the schoolbook decomposition which is known to be valid
    # For 2x3x9, the schoolbook rank is 54.
    # We attempt to find a lower rank decomposition if possible,
    # but fall back to schoolbook if no improvement is found.
    
    # Try a structured approach based on the dimensions
    # n=2, m=3, p=9
    # Since p=9=3*3, we can potentially exploit structure
    
    # First, let's try to see if we can do better than schoolbook
    # using a targeted search for lower rank decompositions
    
    best_result = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try ranks from 53 down to 40 with time budget
    for target_rank in range(53, 39, -1):
        if time.time() > DEADLINE - 5:
            break
        
        result = _try_structured_rank(n, m, p, target_rank)
        if result is not None:
            if _verify(n, m, p, result) == target_rank:
                best_result = result
                best_rank = target_rank
                break  # Found a valid lower rank, return it
    
    return best_result

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, result):
    """Verify that the decomposition is correct. Returns rank if valid, None otherwise."""
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    rank = U.shape[1]
    
    # Check shapes
    if U.shape != (n * m, rank):
        return None
    if V.shape != (m * p, rank):
        return None
    if W.shape != (p * n, rank):
        return None
    
    # Verify: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return None
    
    # Check entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return None
    
    return rank

def _try_structured_rank(n, m, p, target_rank):
    """Try to find a decomposition with the given target rank using structured methods."""
    
    # For 2x3x9, one approach is to use the fact that we can potentially
    # combine some of the schoolbook terms.
    
    # The schoolbook decomposition has 54 terms, one for each (i,j,k) triple.
    # A term corresponds to U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    
    # To reduce rank, we need to find linear dependencies or ways to combine terms.
    
    # One idea: if we can find coefficients a_c such that the sum of scaled
    # rank-1 tensors equals the original tensor with fewer terms.
    
    # For small cases, we can try to solve this as a system of equations.
    
    # Let's try a different approach: use the structure of the dimensions.
    # n=2, m=3, p=9
    
    # We can think of this as multiplying a 2x3 matrix by a 3x9 matrix.
    # The result is a 2x9 matrix.
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # For i=1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # Each C[i,k] is a linear combination of 3 products.
    # There are 18 output elements (2*9), each requiring 3 multiplications.
    # Total: 54 multiplications in schoolbook.
    
    # To reduce the rank, we need to share some of these multiplications.
    
    # One approach: use Winograd-style algorithms or similar techniques.
    
    # For now, let's try a numerical approach:
    # Assume a decomposition of rank r and try to solve for the coefficients.
    
    # This is a non-convex problem, so we'll use random initialization
    # and try to minimize the error.
    
    np.random.seed(42 + target_rank)
    
    # Initialize with random values
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    # Try to optimize using gradient descent
    # Objective: minimize sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    
    learning_rate = 0.01
    num_iterations = 100
    
    for iteration in range(num_iterations):
        if time.time() > DEADLINE - 5:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Current value
                    val = 0.0
                    for c in range(target_rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    
                    error = val - 1.0
                    
                    # Gradient with respect to U[i*m+j, c]
                    for c in range(target_rank):
                        dU[i * m + j, c] += 2 * error * V[j * p + k, c] * W[k * n + i, c]
                        dV[j * p + k, c] += 2 * error * U[i * m + j, c] * W[k * n + i, c]
                        dW[k * n + i, c] += 2 * error * U[i * m + j, c] * V[j * p + k, c]
        
        # Update
        U -= learning_rate * dU
        V -= learning_rate * dV
        W -= learning_rate * dW
    
    # Check if the solution is close to valid
    max_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(target_rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                max_error = max(max_error, abs(val - 1.0))
    
    if max_error < 1e-6:
        # Convert to list and check if entries are close to integer/half-integer
        U_list = [[round(U[i, c] * 2) / 2 for c in range(target_rank)] for i in range(n * m)]
        V_list = [[round(V[i, c] * 2) / 2 for c in range(target_rank)] for i in range(m * p)]
        W_list = [[round(W[i, c] * 2) / 2 for c in range(target_rank)] for i in range(p * n)]
        
        # Verify the rounded solution
        result = {"U": U_list, "V": V_list, "W": W_list}
        if _verify(n, m, p, result) == target_rank:
            return result
    
    return None

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result else 'None'}")
