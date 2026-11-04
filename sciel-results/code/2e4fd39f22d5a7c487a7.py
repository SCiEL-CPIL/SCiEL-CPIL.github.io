"""Rewritten program: search for low-rank exact tensor decomposition of 3x5 * 5x5 multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]
N, M, P = 3, 5, 5
NM, MP, PN = N * M, M * P, P * N  # 15, 25, 15

# Deadline guard: leave safety margin before 150s
DEADLINE = 140.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _remaining():
    return DEADLINE - _elapsed()

def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] for (i,j) in A, k in B."""
    T = np.zeros((NM, MP, PN), dtype=float)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i * M + j, j * P + k, k * N + i] = 1.0
    return T

def _decompose_to_UVW(U, V, W):
    """Given factor matrices U (NM x r), V (MP x r), W (PN x r), return dict."""
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, tol=1e-9):
    """Check if U,V,W exactly decompose the tensor."""
    T = _build_tensor()
    for idx in range(T.shape[0] * T.shape[1] * T.shape[2]):
        i = idx // (MP * PN)
        j = (idx // PN) % MP
        k = idx % PN
        # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        val = 0.0
        for c in range(U.shape[1]):
            val += U[i, c] * V[j, c] * W[k, c]
        if abs(val - T[i, j, k]) > tol:
            return False
    return True

def _verify_fast(U, V, W, tol=1e-9):
    """Faster verification using numpy einsum."""
    T = _build_tensor()
    # Compute reconstructed tensor
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(recon, T, atol=tol)

def _random_init(rank, seed=None):
    """Initialize random factor matrices."""
    rng = np.random.RandomState(seed)
    U = rng.randn(NM, rank)
    V = rng.randn(MP, rank)
    W = rng.randn(PN, rank)
    # Normalize columns to prevent divergence
    for c in range(rank):
        norms = [np.linalg.norm(U[:, c]), np.linalg.norm(V[:, c]), np.linalg.norm(W[:, c])]
        if all(n > 1e-15 for n in norms):
            scale = (norms[0] * norms[1] * norms[2]) ** (1.0 / 3.0)
            U[:, c] /= norms[0]
            V[:, c] /= norms[1]
            W[:, c] /= norms[2]
    return U, V, W

def _gradient_step(U, V, W, lr=0.01, momentum=0.9):
    """One step of gradient descent on the decomposition error."""
    T = _build_tensor()
    
    # Compute gradients
    # Loss = sum_{i,j,k} (sum_c U[i,c]*V[j,c]*W[k,c] - T[i,j,k])^2
    # dL/dU[i,c] = 2 * sum_{j,k} (recon[i,j,k] - T[i,j,k]) * V[j,c] * W[k,c]
    
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    err = recon - T  # (NM, MP, PN)
    
    # Gradient for U: sum over j,k of err[i,j,k] * V[j,c] * W[k,c]
    # G_U[i,c] = sum_{j,k} err[i,j,k] * V[j,c] * W[k,c]
    G_U = np.einsum('ijk,jc,kc->ic', err, V, W)
    G_V = np.einsum('ijk,ic,kc->jc', err, U, W)
    G_W = np.einsum('ijk,ic,jc->kc', err, U, V)
    
    # Gradient descent step
    U -= 2 * lr * G_U
    V -= 2 * lr * G_V
    W -= 2 * lr * G_W
    
    return U, V

def _try_rank(rank, max_iters=2000, lr_start=0.005, lr_end=0.0001, seed_base=42):
    """Try to find a valid decomposition of given rank."""
    T = _build_tensor()
    
    # Try multiple random initializations
    for trial in range(5):
        if _remaining() < 5.0:
            break
        seed = seed_base + trial * 1000
        U, V, W = _random_init(rank, seed=seed)
        
        lr = lr_start
        best_err = float('inf')
        best_UVW = (U.copy(), V.copy(), W.copy())
        
        for it in range(max_iters):
            if _remaining() < 3.0:
                break
            
            # Compute error
            recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = recon - T
            loss = np.sum(err ** 2)
            
            if loss < best_err:
                best_err = loss
                best_UVW = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-10:
                # Found a solution!
                U, V, W = best_UVW
                if _verify_fast(U, V, W, tol=1e-9):
                    return _decompose_to_UVW(U, V, W)
                break
            
            # Gradient step
            G_U = np.einsum('ijk,jc,kc->ic', err, V, W)
            G_V = np.einsum('ijk,ic,kc->jc', err, U, W)
            G_W = np.einsum('ijk,ic,jc->kc', err, U, V)
            
            # Adaptive learning rate
            lr = lr_start * (lr_end / lr_start) ** (it / max_iters)
            
            U -= 2 * lr * G_U
            V -= 2 * lr * G_V
            W -= 2 * lr * G_W
            
            # Renormalize to prevent overflow
            for c in range(rank):
                nU = np.linalg.norm(U[:, c])
                nV = np.linalg.norm(V[:, c])
                nW = np.linalg.norm(W[:, c])
                if nU > 1e-15 and nV > 1e-15 and nW > 1e-15:
                    scale = (nU * nV * nW) ** (1.0 / 3.0)
                    U[:, c] /= nU
                    V[:, c] /= nV
                    W[:, c] /= nW
        
        # Check best solution
        U, V, W = best_UVW
        if _verify_fast(U, V, W, tol=1e-9):
            return _decompose_to_UVW(U, V, W)
    
    return None

def _try_rank_with_structure(rank, max_iters=3000, seed_base=12345):
    """Try rank with structured initialization (sparse patterns)."""
    T = _build_tensor()
    
    for trial in range(3):
        if _remaining() < 5.0:
            break
        seed = seed_base + trial * 7777
        rng = np.random.RandomState(seed)
        
        # Initialize with some structure: random signs and magnitudes
        U = rng.randn(NM, rank) * 0.5
        V = rng.randn(MP, rank) * 0.5
        W = rng.randn(PN, rank) * 0.5
        
        # Add some zeros for sparsity
        for c in range(rank):
            mask = rng.random(NM) < 0.5
            U[mask, c] = 0
            mask = rng.random(MP) < 0.5
            V[mask, c] = 0
            mask = rng.random(PN) < 0.5
            W[mask, c] = 0
        
        lr = 0.01
        best_err = float('inf')
        best_UVW = (U.copy(), V.copy(), W.copy())
        
        for it in range(max_iters):
            if _remaining() < 3.0:
                break
            
            recon = np.einsum('ic,jc,kc->ijk', U, V, W)
            err = recon - T
            loss = np.sum(err ** 2)
            
            if loss < best_err:
                best_err = loss
                best_UVW = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-10:
                U, V, W = best_UVW
                if _verify_fast(U, V, W, tol=1e-9):
                    return _decompose_to_UVW(U, V, W)
                break
            
            G_U = np.einsum('ijk,jc,kc->ic', err, V, W)
            G_V = np.einsum('ijk,ic,kc->jc', err, U, W)
            G_W = np.einsum('ijk,ic,jc->kc', err, U, V)
            
            U -= 2 * lr * G_U
            V -= 2 * lr * G_V
            W -= 2 * lr * G_W
            
            # Decay learning rate
            if it % 100 == 0 and it > 0:
                lr *= 0.95
        
        U, V, W = best_UVW
        if _verify_fast(U, V, W, tol=1e-9):
            return _decompose_to_UVW(U, V, W)
    
    return None

def _schoolbook():
    """Return the schoolbook (rank 75) decomposition as fallback."""
    rank = N * M * P
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x5 matrix multiplication."""
    global _start_time
    _start_time = time.time()
    
    # Try decreasing ranks from a reasonable lower bound
    # Lower bound: rank >= max(NM, MP, PN) = 25
    # Try ranks from 25 upward, but limit total search time
    
    ranks_to_try = list(range(25, 76))  # 25 to 75
    
    for rank in ranks_to_try:
        if _remaining() < 10.0:
            break
        
        # Try standard gradient descent
        result = _try_rank(rank, max_iters=1500, lr_start=0.005, lr_end=0.0001, seed_base=rank * 100)
        if result is not None:
            return result
        
        if _remaining() < 10.0:
            break
        
        # Try structured initialization
        result = _try_rank_with_structure(rank, max_iters=2000, seed_base=rank * 777)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook()
