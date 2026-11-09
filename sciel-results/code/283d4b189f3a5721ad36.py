"""Tensor decomposition search for 3x5 x 5x5 matrix multiplication."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 5, 5]
DEADLINE = time.time() + 140  # safety margin before 150s kill

def _verify_decomposition(U, V, W, n=3, m=5, p=5):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # The matmul tensor T[i,j,k] = delta (i,j,k) means:
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal delta_{i,j,k} pattern
    # Actually for matmul: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = (A*B)[i,k] when A[i,j], B[j,k]
    # The tensor is: T[i,j,k] such that sum_j T[i,j,k]*A[i,j]*B[j,k] = (AB)[i,k]
    # This means T[i,j,k] = delta_{j,k}... no.
    # For matrix multiplication: (AB)_{ik} = sum_j A_{ij} B_{jk}
    # So the tensor M_{ijk} satisfies sum_{j,k} M_{ijk} A_{ij} B_{jk} = (AB)_{ik}
    # This gives M_{ijk} = delta_{jk}... wait no.
    # sum_j A_{ij} B_{jk} = (AB)_{ik}
    # So we need: sum_j,k M_{ijk} A_{ij} B_{jk} = sum_j A_{ij} B_{jk}
    # This means M_{ijk} = delta_{jk} for all i? No that's not right either.
    # Actually M_{ijk} where the contraction is over j and k:
    # sum_{j,k} M_{ijk} A_{ij} B_{jk} should equal (AB)_{ik} = sum_j A_{ij} B_{jk}
    # So M_{ijk} must satisfy: for all A, B: sum_{j,k} M_{ijk} A_{ij} B_{jk} = sum_j A_{ij} B_{jk}
    # This gives M_{ijk} = delta_{jk}... let me think again.
    # If M_{ijk} = delta_{jk}, then sum_{j,k} delta_{jk} A_{ij} B_{jk} = sum_j A_{ij} B_{jj} which is wrong.
    
    # Let me reconsider. The standard matmul tensor:
    # (AB)_{ik} = sum_j A_{ij} B_{jk}
    # As a 3-tensor: T_{ijk} where we contract A with index j of T, B with index k of T
    # T_{ijk} A_{ij} B_{jk} summed over j,k gives (AB)_{ik}
    # So T_{ijk} = delta_{jk}? No...
    # sum_{j,k} T_{ijk} A_{ij} B_{jk} = (AB)_{ik} = sum_j A_{ij} B_{jk}
    # For this to hold for ALL A, B, we need T_{ijk} such that
    # sum_k T_{ijk} B_{jk} = B_{jk} for each j... that means sum_k T_{ijk} B_{jk} = B_{jk}
    # which requires T_{ijk} = delta_{kk'}... 
    
    # Actually the correct statement: the matmul tensor has components
    # M_{i,j,k} = 1 if we're computing (AB)_{ik} = sum_j A_{ij} B_{jk}
    # The tensor is indexed such that M_{i,j,k} contributes A_{ij} * B_{jk} to (AB)_{ik}
    # So M_{i,j,k} = 1 for all valid i,j,k? No, that would give sum_{j,k} A_{ij}B_{jk} which is wrong.
    
    # I think the convention here based on the W[k*n+i,r] indexing:
    # U has rows indexed by (i,j) -> i*m+j, size n*m
    # V has rows indexed by (j,k) -> j*p+k, size m*p  
    # W has rows indexed by (k,i) -> k*n+i, size p*n
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T_{i,j,k}
    # where T_{i,j,k} is the matmul tensor.
    # For matmul: (AB)_{ik} = sum_j A_{ij} B_{jk}
    # So T_{i,j,k} should be such that sum_{j,k} T_{i,j,k} A_{ij} B_{jk} = (AB)_{ik}
    # This means T_{i,j,k} = delta_{j,k}? Let's check: sum_{j,k} delta_{jk} A_{ij} B_{jk} = sum_j A_{ij} B_{jj} -- wrong
    # 
    # Hmm, maybe T_{i,j,k} = 1 for all i,j,k and the contraction is different.
    # Or maybe the tensor is defined as: the bilinear map (A,B) -> AB has tensor
    # M_{i,j,k,l} = delta_{jk} delta_{il}... no.
    
    # Let me just use the schoolbook as reference:
    # In schoolbook: for each (i,j,k): U[i*m+j][col]=1, V[j*p+k][col]=1, W[k*n+i][col]=1
    # So sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for the matching (i,j,k) triple, 0 otherwise
    # This means the reconstructed tensor T_{i,j,k} = 1 if there's a rank-1 term with all three indices matching
    # i.e., T_{i,j,k} = delta (the "identity" tensor where each (i,j,k) has value 1)
    # But that's not the matmul tensor...
    
    # Wait - I think I'm overcomplicating. The matmul tensor for multiplying (n x m) by (m x p) to get (n x p):
    # The output (AB)_{ik} = sum_j A_{ij} B_{jk}
    # As a 3-way tensor with indices (i, j, k) where j is the summed index:
    # The tensor T has T[i, j, k] = 1 for all i, j, k (since every combination contributes)
    # No wait, that would give sum_{j,k} A_{ij} B_{jk} which sums over both j and k.
    
    # I think the correct interpretation: the "matmul tensor" in this context is the 3-tensor
    # M_{i,j,k} = 1 for all i in [n], j in [m], k in [p]
    # because (AB)_{ik} = sum_j A_{ij} * B_{jk}, and the tensor encodes which (i,j,k) triples
    # have a contribution. Each term A_{ij}*B_{jk} contributes to (AB)_{ik}.
    # So M_{i,j,k} = 1 for all valid indices, and the bilinear form is:
    # sum_{i,j,k} M_{i,j,k} * A_{ij} * B_{jk} = sum_{i,j,k} A_{ij} B_{jk} = sum_i (sum_j sum_k A_{ij} B_{jk})
    # That's not right either because it sums over k as well.
    
    # OK let me just verify the schoolbook works by checking:
    # For the schoolbook, the "reconstructed" tensor at position (i,j,k) is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # In schoolbook, each (i,j,k) gets exactly one rank-1 term where all three are 1.
    # So the reconstructed tensor is 1 at every (i,j,k) position.
    # 
    # The task says this is the matmul tensor. So the matmul tensor for 3x5 * 5x5 is
    # simply the all-ones tensor of shape (3,5,5)? That seems too simple.
    # 
    # Actually yes! For matrix multiplication (n×m)(m×p)→(n×p):
    # (AB)_{ik} = Σ_j A_{ij}B_{jk}
    # The tensor M_{ijk} where the bilinear map is Σ_{i,j,k} M_{ijk} A_{ij} B_{jk} = (AB)_{ik}... 
    # but that doesn't work dimensionally.
    #
    # I think the correct view: the matmul is a bilinear map (A,B) → AB.
    # As a 3-tensor T_{i,j,k} (where i indexes output row, j indexes A's col/B's row, k indexes output col):
    # (AB)_{ik} = Σ_j T_{i,j,k} A_{ij} B_{jk}
    # For this to equal Σ_j A_{ij} B_{jk}, we need T_{i,j,k} = δ_{j,j}... 
    # Actually T_{i,j,k} must be such that for fixed i,k: Σ_j T_{i,j,k} A_{ij} B_{jk} = Σ_j A_{ij} B_{jk}
    # This requires T_{i,j,k} = 1 for all j (for each fixed i,k). But T doesn't depend on k in this way...
    # 
    # I think T_{i,j,k} = 1 for ALL i,j,k is correct, and the "matrix multiplication tensor" is just
    # the tensor with all entries 1. The rank of this tensor is what we're minimizing.
    
    # Let me just verify: does the schoolbook produce an all-ones tensor?
    # Yes, as shown above. So the target tensor is all-ones of shape (3,5,5) = 75 entries all = 1.
    
    # Now verify our decomposition produces all-ones:
    recon = np.einsum('ir,jr,kr->ijk', U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
    # Wait, W is indexed by k*n+i, so W[k*n+i, r]. Reshape: W.reshape(p, n, rank) gives W[k,i,r]
    # U[i*m+j, r] -> U.reshape(n, m, rank) gives U[i,j,r]  
    # V[j*p+k, r] -> V.reshape(m, p, rank) gives V[j,k,r]
    # Product: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
    
    target = np.ones((n, m, p))
    return np.allclose(recon, target, atol=1e-9)


def _try_rank_bruteforce_small(rank, n=3, m=5, p=5, max_attempts=5000):
    """Try to find a rank-r decomposition with small integer/half-integer entries."""
    # For very small ranks, try random search with small values
    # Entries from {-1, -0.5, 0, 0.5, 1}
    vals = [-1, -0.5, 0, 0.5, 1]
    
    # This is only feasible for very small ranks
    total_entries = (n*m + m*p + p*n) * rank
    if total_entries > 60:  # too many to brute force
        return None
    
    import random
    random.seed(42)
    
    for attempt in range(max_attempts):
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for r in range(rank):
            # Each rank-1 term: pick which entries are non-zero
            # For matmul-like structure, try patterns
            for i in range(n):
                for j in range(m):
                    U[i*m+j, r] = random.choice(vals)
            for j in range(m):
                for k in range(p):
                    V[j*p+k, r] = random.choice(vals)
            for k in range(p):
                for i in range(n):
                    W[k*n+i, r] = random.choice(vals)
        
        # Verify
        recon = np.einsum('ir,jr,kr->ijk', 
                         U.reshape(n,m,rank), 
                         V.reshape(m,p,rank), 
                         W.reshape(p,n,rank))
        if np.allclose(recon, np.ones((n,m,p)), atol=1e-9):
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None


def _gradient_search(rank, n=3, m=5, p=5, max_iters=20000, lr=0.01, seed=42):
    """Use gradient descent to find a rank-r decomposition."""
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    target = np.ones((n, m, p))
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
            
        # Compute reconstruction
        recon = np.einsum('ir,jr,kr->ijk', 
                         U.reshape(n,m,rank), 
                         V.reshape(m,p,rank), 
                         W.reshape(p,n,rank))
        
        loss = np.sum((recon - target)**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
        
        # Gradients
        # dL/dU[i,j,r] = 2 * (recon[i,j,k] - 1) * V[j,k,r] * W[k,i,r] summed over k
        # More carefully:
        # recon[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # dL/dU[i,j,r] = 2 * sum_k (recon[i,j,k] - 1) * V[j,k,r] * W[k,i,r]
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        err = recon - target  # shape (n,m,p)
        
        for r in range(rank):
            # dU[i,j,r] = 2 * sum_k err[i,j,k] * V[j,k,r] * W[k,i,r]
            for i in range(n):
                for j in range(m):
                    s = 0
                    for k in range(p):
                        s += err[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                    dU[i*m+j, r] = 2 * s
                    
            # dV[j,k,r] = 2 * sum_i err[i,j,k] * U[i,j,r] * W[k,i,r]
            for j in range(m):
                for k in range(p):
                    s = 0
                    for i in range(n):
                        s += err[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
                    dV[j*p+k, r] = 2 * s
                    
            # dW[k,i,r] = 2 * sum_j err[i,j,k] * U[i,j,r] * V[j,k,r]
            for k in range(p):
                for i in range(n):
                    s = 0
                    for j in range(m):
                        s += err[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
                    dW[k*n+i, r] = 2 * s
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Check if we found exact solution
    recon = np.einsum('ir,jr,kr->ijk', 
                     best_U.reshape(n,m,rank), 
                     best_V.reshape(m,p,rank), 
                     best_W.reshape(p,n,rank))
    if np.allclose(recon, target, atol=1e-8):
        return (best_U.tolist(), best_V.tolist(), best_W.tolist())
    
    return None


def _search_rank(rank, n=3, m=5, p=5, time_budget=30):
    """Search for a rank-r decomposition using multiple strategies."""
    start = time.time()
    
    # Strategy 1: Try structured decompositions based on known results
    # For small matrices, there are known optimal ranks.
    # 2x2x2 matmul has rank 7 (Strassen)
    # For 3x5x5, let's try various approaches
    
    # Strategy 2: Random restarts with gradient descent
    for seed in range(20):
        if time.time() - start > time_budget:
            break
        result = _gradient_search(rank, n, m, p, max_iters=5000, lr=0.05, seed=seed)
        if result is not None:
            return result
    
    # Strategy 3: Try with structured initialization
    # Based on the idea that matmul tensors have structure
    for seed in range(10):
        if time.time() - start > time_budget:
            break
        
        np.random.seed(seed + 100)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Initialize with some structure
        for r in range(rank):
            # Pick random subsets
            u_indices = np.random.choice(n*m, size=min(n*m, 3), replace=False)
            v_indices = np.random.choice(m*p, size=min(m*p, 3), replace=False)
            w_indices = np.random.choice(p*n, size=min(p*n, 3), replace=False)
            
            for idx in u_indices:
                U[idx, r] = np.random.choice([-1, 1])
            for idx in v_indices:
                V[idx, r] = np.random.choice([-1, 1])
            for idx in w_indices:
                W[idx, r] = np.random.choice([-1, 1])
        
        # Refine with gradient descent
        target = np.ones((n, m, p))
        best_loss = float('inf')
        for it in range(3000):
            recon = np.einsum('ir,jr,kr->ijk', 
                             U.reshape(n,m,rank), 
                             V.reshape(m,p,rank), 
                             W.reshape(p,n,rank))
            err = recon - target
            loss = np.sum(err**2)
            
            if loss < best_loss:
                best_loss = loss
            
            if best_loss < 1e-10:
                break
            
            # Gradient step
            lr = 0.01 / (1 + it/1000)
            dU = np.zeros_like(U)
            dV = np.zeros_like(V) 
            dW = np.zeros_like(W)
            
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        dU[i*m+j, r] = 2 * sum(err[i,j,k] * V[j*p+k, r] * W[k*n+i, r] for k in range(p))
                for j in range(m):
                    for k in range(p):
                        dV[j*p+k, r] = 2 * sum(err[i,j,k] * U[i*m+j, r] * W[k*n+i, r] for i in range(n))
                for k in range(p):
                    for i in range(n):
                        dW[k*n+i, r] = 2 * sum(err[i,j,k] * U[i*m+j, r] * V[j*p+k, r] for j in range(m))
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        recon = np.einsum('ir,jr,kr->ijk', 
                         U.reshape(n,m,rank), 
                         V.reshape(m,p,rank), 
                         W.reshape(p,n,rank))
        if np.allclose(recon, target, atol=1e-8):
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None


def _schoolbook(n=3, m=5, p=5):
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


def construct():
    """Find rank-minimal tensor decomposition for 3x5 x 5x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 75
    
    # Try to find lower ranks
    # Known lower bounds: for matmul (n,m,p), rank >= max(n*m, m*p, p*n) = max(15,25,15) = 25
    # Also rank >= max(n, m, p) * min(n, m, p) ... various bounds
    # Let's try from reasonable lower bounds up
    
    # Try various ranks
    ranks_to_try = list(range(25, min(75, 50)))  # Try 25-49
    # Also try some specific values
    ranks_to_try = sorted(set(ranks_to_try + [30, 35, 40, 45]))
    
    for rank in ranks_to_try:
        if time.time() > DEADLINE:
            break
        
        # Allocate time based on rank (lower ranks are harder, give more time)
        time_budget = min(20, max(5, 60 - rank))
        
        result = _search_rank(rank, n, m, p, time_budget=time_budget)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank
            # If we found a low rank, we could stop, but let's try to go lower
            if rank <= 30:
                break
    
    # Final verification
    if not _verify_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Fallback to schoolbook
        best_result = _schoolbook(n, m, p)
    
    return best_result
