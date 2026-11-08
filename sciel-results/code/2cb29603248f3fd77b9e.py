a_part = np.sum(Ur * A_test, axis=1)  # a_part[i] = Σ_j U[i,j,r] * A[i,j]
b_part = np.sum(Vr * B_test, axis=0)  # b_part[k] = Σ_j V[j,k,r] * B[j,k]
C_computed[i,k] += a_part[i] * b_part[k] * Wr[k,i]
