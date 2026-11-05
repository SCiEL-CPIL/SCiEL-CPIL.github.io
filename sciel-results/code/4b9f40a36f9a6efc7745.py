import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Compute the graph Laplacian L = D - A.
    2. Find the Fiedler vector (eigenvector for the 2nd smallest eigenvalue).
       For disconnected graphs, we might need to handle multiple zero eigenvalues,
       but for ER(200, p) with reasonable p, it's likely connected. 
       If disconnected, the "second smallest" might be 0. We want the vector that
       provides the best partition, which is typically the one with the smallest
       non-zero eigenvalue (algebraic connectivity). 
       scipy.sparse.linalg.eigsh with k=2 gives the two smallest. 
       The first is usually the constant vector (eigenvalue 0) if connected.
       The second is the Fiedler vector.
       
    3. Score vertices by their absolute value in the Fiedler vector.
       Rationale: Vertices with large |fiedler[i]| are far from the "center" of the
       spectral partition. In the context of MIS, selecting vertices that are
       spectrally distinct or on the periphery of the main component might help
       in forming a large independent set by avoiding the dense, highly connected
       core where vertices compete for selection. 
       
       Alternatively, we can look at this as a relaxation of the Max-Cut problem.
       MIS is related to Max-Cut. The Fiedler vector provides a spectral relaxation
       of Max-Cut. Vertices with high absolute values are those that are strongly
       "assigned" to one side of the cut. In a greedy MIS decoder, picking a vertex
       removes it and its neighbors. If we pick vertices that are spectrally extreme,
       we might be effectively performing a greedy approximation of the spectral
       partition, which tends to yield large independent sets (one side of the cut
       is an independent set in the bipartite subgraph, or close to it).
       
       Let's try: score[i] = abs(fiedler[i])
       
    4. Fallback to degree-based if spectral computation fails or times out.
    """
    start_time = time.time()
    DEADLINE = 100.0  # Safety margin before 120s limit
    
    try:
        # Convert to float matrix
        adj = np.asarray(adjacency, dtype=float)
        n = adj.shape[0]
        
        # Check for immediate timeout
        if time.time() - start_time > DEADLINE:
            return -np.sum(adj, axis=1)
            
        # Compute degrees
        deg = adj.sum(axis=1)
        
        # If graph is empty or trivial, return simple score
        if n < 2:
            return np.zeros(n)
        
        # Build Laplacian L = D - A
        # D is diagonal matrix of degrees
        # We can construct L directly as sparse or dense. 
        # For n=200, dense is fine and fast, but sparse is safer for generalization.
        # Let's use scipy.sparse for robustness.
        
        # L = D - A
        # In sparse format:
        # D is diag(deg)
        # A is the adjacency matrix
        
        # Create sparse adjacency
        A_sp = csr_matrix(adj)
        
        # Create diagonal degree matrix
        D_sp = csr_matrix((deg, (np.arange(n), np.arange(n))))
        
        # Laplacian
        L = D_sp - A_sp
        
        # Find the 2 smallest eigenvalues and their eigenvectors
        # k=2, sigma=None (smallest magnitude)
        # If the graph is disconnected, the smallest eigenvalue is 0 with multiplicity > 1.
        # eigsh might return a vector in the null space. 
        # We want the eigenvector associated with the smallest positive eigenvalue.
        # For connected graphs, lambda_1 = 0, lambda_2 > 0.
        # For disconnected, lambda_1 = ... = 0.
        
        try:
            # k=2, return_eigenvectors=True
            # v0 is the smallest, v1 is the second smallest.
            # For connected graph: v0 is constant, v1 is Fiedler.
            # For disconnected graph: v0 and v1 might both be in the null space.
            # To be robust, we can check eigenvalues.
            
            eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')
            
            # Sort by eigenvalue just in case
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # If the second smallest eigenvalue is close to 0, the graph is likely disconnected.
            # In that case, the Fiedler vector concept is less clear. 
            # However, for ER graphs with n=200, if p is small enough to disconnect,
            # the components are small. 
            # If lambda_1 < 1e-8 and lambda_2 < 1e-8, we are in the null space.
            # We should pick a vector that actually partitions the graph.
            # A common trick: if there are multiple zero eigenvalues, pick the eigenvector
            # with the largest variance or norm? 
            # Actually, if the graph is disconnected, we can just solve MIS on each component.
            # But the greedy decoder is global. 
            # Let's stick to the Fiedler vector of the largest component or just use the 
            # second eigenvector returned. If it's in the null space, it might be localized.
            
            # For now, assume connected or use the second eigenvector returned.
            fiedler = eigenvectors[:, 1]
            
            # Normalize to avoid scaling issues (though score ordering is invariant to scaling)
            # abs(fiedler) gives the "spectral distance" from the center.
            
            # Check if the vector is useful (not all zeros)
            if np.linalg.norm(fiedler) < 1e-10:
                raise ValueError("Fiedler vector is zero")
                
            score = np.abs(fiedler)
            
        except Exception:
            # Fallback to degree-based score if spectral method fails
            if time.time() - start_time > DEADLINE:
                return -deg
            score = -deg
            
    except Exception:
        # Fallback to simple degree score
        try:
            deg = np.sum(adjacency, axis=1)
            score = -deg
        except:
            score = np.zeros(adjacency.shape[0])
            
    if time.time() - start_time > DEADLINE:
        # Final safety check
        try:
            return -np.sum(adjacency, axis=1)
        except:
            return np.zeros(adjacency.shape[0])
            
    return score
