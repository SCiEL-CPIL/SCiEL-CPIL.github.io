# U: (nm, r) = (6, r), reshape to (2, 3, r)
# V: (mp, r) = (12, r), reshape to (3, 4, r)
# W: (pn, r) = (8, r), reshape to (4, 2, r)
# Result: (2, 3, 3, 4, 4, 2)
# T[ia, ja, jb, kb, k, i] = sum_r W[k, i, r] * U[ia, ja, r] * V[jb, kb, r]
T = np.einsum('kir,ijr,jkr->ijkkl'... 
