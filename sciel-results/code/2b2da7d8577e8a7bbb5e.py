import numpy as np
import random
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    
    # We search for a lower rank
    # Lower bound heuristic: max(n*m, m*p, n*p) is not a lower bound for rank, 
    # but rank >= max(n, m, p) is true? No.
    # Rank of M_{2,2,2} is 7. max(2,2,2)=2.
    # Let's try R from 15 up to rank_school - 1
    # If we find one, return it.
    
    # To speed up, we can try known good ranks first or just scan.
    # Scanning from low to high is better for finding minimal.
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s margin
    
    # Try ranks from 12 to 59
    # 12 is a theoretical lower bound ceil(60/5)
    for R in range(12, rank_school):
        if time.time() - start_time > timeout:
            break
            
        found = False
        # Try a few random initializations
        # Using small integers for U and V
        for attempt in range(200):
            if time.time() - start_time > timeout:
                break
                
            # Random U and V with entries in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n, R))
            V = np.random.randint(-1, 2, size=(m, R))
            
            # Construct A_stack
            # A_j[i, r] = U[i, r] * V[j, r]
            # We need A_j W = J for all j
            # Stacked: A_stack[k, r] where k = j*n + i
            A_stack = np.zeros((m * n, R))
            b = np.ones(m * n)
            
            for j in range(m):
                for i in range(n):
                    idx = j * n + i
                    A_stack[idx, :] = U[i, :] * V[j, :]
            
            # Check consistency of A_stack x = b
            # Using least squares to check residual
            try:
                sol, residuals, rank, s = np.linalg.lstsq(A_stack, b, rcond=None)
                # Check if residual is close to 0
                if residuals.size > 0 and residuals[0] < 1e-9:
                    # Consistent!
                    W = sol.reshape(R, p) # Wait, sol is R x 1?
                    # b is m*n x 1.
                    # We need p columns of W.
                    # The system A_stack x = b finds ONE x that works for ALL i,j.
                    # This x corresponds to ONE column of W.
                    # We need p columns.
                    # Since the system is linear, any solution x works.
                    # We can just use the same x for all p columns?
                    # If W = [x, x, ..., x], then A_j W = [J, J, ..., J] = J_{n x p}.
                    # Yes, this works.
                    W_full = np.tile(sol.reshape(R, 1), (1, p))
                    
                    # Verify
                    # M_ijk = sum_r U[i,r] V[j,r] W[r,k]
                    # Let's check a few entries or all
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = sum(U[i, r] * V[j, r] * W_full[r, k] for r in range(R))
                                if abs(val - 1.0) > 1e-6:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    
                    if valid:
                        # Convert to rational/integer
                        # sol might be float.
                        # We can scale U, V, W to integers.
                        # Let's try to find a common denominator.
                        # For now, just return floats? No, must be integer/half-integer.
                        # We can multiply by a large number and round?
                        # Or use Fraction.
                        
                        # Simple approach: 
                        # If U, V are integers, and W is rational, 
                        # we can scale W to make it integer.
                        # But U, V are integers.
                        # The product U*V*W must be integer 1.
                        # If W has denominators, the sum must cancel them.
                        # Let's try to find integer W.
                        # We can solve for W in integers?
                        # A_stack is integer. b is integer.
                        # If a rational solution exists, an integer solution might not.
                        # But we can scale the whole decomposition.
                        # If (U, V, W) is a solution, (cU, V, W/c) is not necessarily.
                        # (U, V, cW) -> c*M. No.
                        # We need M=1.
                        
                        # Let's use Fraction to clean up.
                        from fractions import Fraction
                        U_frac = [[Fraction(U[i, r]).limit_denominator(1000) for r in range(R)] for i in range(n)]
                        V_frac = [[Fraction(V[j, r]).limit_denominator(1000) for r in range(R)] for j in range(m)]
                        W_frac = [[Fraction(W_full[r, k]).limit_denominator(1000) for k in range(p)] for r in range(R)]
                        
                        # Check if they are close to original
                        # If yes, use them.
                        
                        # Convert to list of lists for output
                        U_out = [[float(x) for x in row] for row in U_frac]
                        V_out = [[float(x) for x in row] for row in V_frac]
                        W_out = [[float(x) for x in row] for row in W_frac]
                        
                        # The prompt asks for U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank)
                        # Wait, the interface says:
                        # U: (n*m, rank)
                        # V: (m*p, rank)
                        # W: (p*n, rank)
                        # My U is (n, R). I need to reshape.
                        # The tensor indices are i, j, k.
                        # U corresponds to i.
                        # V corresponds to j.
                        # W corresponds to k.
                        # But the interface expects:
                        # U[i*m+j, c] ? No.
                        # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]" in the seed.
                        # This implies the factors are mixed?
                        # Let's re-read the seed.
                        # Seed:
                        # U[i * m + j][column] = 1
                        # V[j * p + k][column] = 1
                        # W[k * n + i][column] = 1
                        # This is NOT a standard rank-1 decomposition $u \otimes v \otimes w$.
                        # This is a decomposition of the *permutation* or something?
                        # No, the seed is the schoolbook algorithm.
                        # In schoolbook, for each (i, j, k), we have a term.
                        # The term is $A_{ij} B_{jk}$.
                        # The result is $C_{ik}$.
                        # The seed constructs U, V, W such that:
                        # $C_{ik} = \sum_{r} U_{\text{row}_U(r), r} V_{\text{row}_V(r), r} W_{\text{row}_W(r), r}$?
                        # No, the standard way to represent bilinear maps with U, V, W matrices is:
                        # $C = \sum_r u_r v_r^T w_r^T$?
                        # Let's look at the shapes.
                        # U is (n*m, R).
                        # V is (m*p, R).
                        # W is (p*n, R).
                        # This looks like the "3-way" representation where the indices are flattened.
                        # Actually, for $M_{n,m,p}$, the tensor has dimensions $n \times m \times p$.
                        # A rank-1 tensor is $u \otimes v \otimes w$.
                        # If we flatten the indices, we get vectors of size $nmp$.
                        # But here the sizes are $nm, mp, pn$.
                        # This corresponds to the mode-1, mode-2, mode-3 unfoldings?
                        # No.
                        # Let's look at the seed's logic.
                        # It sets U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
                        # This implies that for a specific (i,j,k), the r-th term is non-zero only at these specific indices.
                        # This is exactly the schoolbook decomposition.
                        # The bilinear form is $\sum_r U_{\alpha r} V_{\beta r} W_{\gamma r}$.
                        # Here $\alpha = i \cdot m + j$, $\beta = j \cdot p + k$, $\gamma = k \cdot n + i$.
                        # This does NOT look like $A_{ij} B_{jk}$.
                        # Wait, if I sum over r:
                        # $\sum_r U_{i m + j, r} V_{j p + k, r} W_{k n + i, r}$.
                        # In the seed, for a fixed r corresponding to (i', j', k'),
                        # U is 1 only if $i=i', j=j'$.
                        # V is 1 only if $j=j', k=k'$.
                        # W is 1 only if $k=k', i=i'$.
                        # So the product is 1 only if $i=i', j=j', k=k'$ AND $j=j'$ AND $k=k'$ AND $i=i'$.
                        # So it is 1 if $(i,j,k) = (i',j',k')$.
                        # So the sum is 1 if there is an r such that $i=i', j=j', k=k'$.
                        # Since r maps to (i,j,k) uniquely, the sum is 1 for all (i,j,k).
                        # So it computes the tensor of all ones.
                        # But matrix multiplication tensor $M_{ijk}$ is 1 for all i,j,k.
                        # So the seed is correct.
                        
                        # So my U, V, W must be reshaped to match these indices.
                        # My U is $n \times R$. I need $n \cdot m \times R$.
                        # My V is $m \times R$. I need $m \cdot p \times R$.
                        # My W is $p \times R$. I need $p \cdot n \times R$.
                        
                        # How to map?
                        # The seed uses $U_{i m + j}$.
                        # My U has index $i$.
                        # I need to replicate my U for each j?
                        # No, the decomposition is $\sum_r u_{r,i} v_{r,j} w_{r,k}$.
                        # The term in the sum is $u_{r,i} v_{r,j} w_{r,k}$.
                        # The interface expects matrices U, V, W such that:
                        # $\sum_r U_{\alpha r} V_{\beta r} W_{\gamma r} = M_{\alpha, \beta, \gamma}$?
                        # No, the indices $\alpha, \beta, \gamma$ are not independent in the seed's mapping.
                        # $\alpha = im+j, \beta = jp+k, \gamma = kn+i$.
                        # This is a specific embedding.
                        
                        # I should generate U, V, W in the format required by the interface.
                        # Let's define:
                        # $U_{im+j, r} = u_{r,i}$
                        # $V_{jp+k, r} = v_{r,j}$
                        # $W_{kn+i, r} = w_{r,k}$
                        
                        # Then $\sum_r U_{im+j, r} V_{jp+k, r} W_{kn+i, r} = \sum_r u_{r,i} v_{r,j} w_{r,k} = M_{ijk}$.
                        # This matches the requirement.
                        
                        # So I just need to expand my U, V, W.
                        
                        U_expanded = np.zeros((n*m, R))
                        for i in range(n):
                            for j in range(m):
                                U_expanded[i*m + j, :] = U[i, :]
                                
                        V_expanded = np.zeros((m*p, R))
                        for j in range(m):
                            for k in range(p):
                                V_expanded[j*p + k, :] = V[j, :]
                                
                        W_expanded = np.zeros((p*n, R))
                        for k in range(p):
                            for i in range(n):
                                W_expanded[k*n + i, :] = W_full[:, k] # Wait, W_full is R x p. W_full[r, k] is w_{r,k}.
                                # My W_full[r, k] corresponds to w_{r,k}.
                                # So W_expanded[k*n+i, r] should be w_{r,k}.
                                # So W_expanded[:, r] for fixed k?
                                # W_expanded is (p*n, R).
                                # Index is k*n+i.
                                # Value is w_{r,k}.
                                # So for each r, W_expanded[k*n+i, r] = W_full[r, k].
                                pass
                                
                        # Let's do it properly
                        W_expanded = np.zeros((p*n, R))
                        for k in range(p):
                            for i in range(n):
                                for r in range(R):
                                    W_expanded[k*n + i, r] = W_full[r, k]
                                    
                        # Return
                        return {
                            "U": U_expanded.tolist(),
                            "V": V_expanded.tolist(),
                            "W": W_expanded.tolist()
                        }
                        
            except np.linalg.LinAlgError:
                continue
                
    # If no better found, return schoolbook
    rank = rank_school
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
# EVOLVE-BLOCK-END
