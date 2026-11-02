def _find_linear_dependencies(U, V, W):
    from fractions import Fraction
    import numpy as np
    
    Uf = np.array(U, dtype=object)
    Vf = np.array(V, dtype=object)
    Wf = np.array(W, dtype=object)
    n_terms = Uf.shape[1]
    
    # Convert to Fraction
    Uf = [[Fraction(x) for x in row] for row in Uf.tolist()]
    Vf = [[Fraction(x) for x in row] for row in Vf.tolist()]
    Wf = [[Fraction(x) for x in row] for row in Wf.tolist()]
    
    removed = set()
    
    # Check pairs first (already done in _exact_rank_reduce, but let's be safe)
    # Then check triplets
    for i in range(n_terms):
        if i in removed: continue
        for j in range(i+1, n_terms):
            if j in removed: continue
            for k in range(j+1, n_terms):
                if k in removed: continue
                
                # Check if col_k = a*col_i + b*col_j
                # We need to find a, b such that:
                # U_k = a U_i + b U_j
                # V_k = a V_i + b V_j
                # W_k = a W_i + b W_j
                
                # Pick a non-zero entry in U_i to determine a or b?
                # Better: pick two non-zero entries in U_i or U_j to solve for a, b.
                # If U_i and U_j are linearly independent in U, we can solve for a, b using U.
                # Then verify with V and W.
                
                # Find indices where U_i or U_j is non-zero
                # Let's just solve the linear system for a, b using all rows of U.
                # This is over-determined. We can use least squares or just pick 2 rows.
                
                # Pick 2 rows r1, r2 such that the 2x2 matrix [[U_i[r1], U_j[r1]], [U_i[r2], U_j[r2]]] is invertible.
                found = False
                a = None
                b = None
                for r1 in range(len(Uf)):
                    if found: break
                    for r2 in range(r1+1, len(Uf)):
                        det = Uf[r1][i] * Uf[r2][j] - Uf[r1][j] * Uf[r2][i]
                        if det != 0:
                            # Solve
                            # a U_i[r1] + b U_j[r1] = U_k[r1]
                            # a U_i[r2] + b U_j[r2] = U_k[r2]
                            inv_det = 1 / det
                            a = (Uf[r1][k] * Uf[r2][j] - Uf[r2][k] * Uf[r1][j]) * inv_det
                            b = (Uf[r2][k] * Uf[r1][i] - Uf[r1][k] * Uf[r2][i]) * inv_det
                            found = True
                            break
                
                if not found:
                    continue
                    
                # Verify with U (all rows)
                ok = True
                for r in range(len(Uf)):
                    if a * Uf[r][i] + b * Uf[r][j] != Uf[r][k]:
                        ok = False
                        break
                if not ok: continue
                
                # Verify with V
                for r in range(len(Vf)):
                    if a * Vf[r][i] + b * Vf[r][j] != Vf[r][k]:
                        ok = False
                        break
                if not ok: continue
                
                # Verify with W
                for r in range(len(Wf)):
                    if a * Wf[r][i] + b * Wf[r][j] != Wf[r][k]:
                        ok = False
                        break
                if not ok: continue
                
                # If ok, remove k
                removed.add(k)
                
    if removed:
        keep = [j for j in range(n_terms) if j not in removed]
        Uf = [[Uf[i][j] for j in keep] for i in range(len(Uf))]
        Vf = [[Vf[i][j] for j in keep] for i in range(len(Vf))]
        Wf = [[Wf[i][j] for j in keep] for i in range(len(Wf))]
        
    # Convert back
    def to_python(lst):
        out = []
        for row in lst:
            new_row = []
            for val in row:
                if val.denominator == 1:
                    new_row.append(val.numerator)
                else:
                    new_row.append(float(val))
            out.append(new_row)
        return out
        
    return to_python(Uf), to_python(Vf), to_python(Wf)
