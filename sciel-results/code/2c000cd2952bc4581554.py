    # Pre-compute constant matrices
    a = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    n_minus_1 = n - np.eye(3)
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)
    
    # Pre-compute terms for Liouvillian
    # L_coherent = -i (H \otimes I - I \otimes H^T)
    # L_incoherent is constant part + H dependent part?
    # No, L_incoherent depends on L1, L2 which are constant.
    # So L_incoherent is a constant matrix!
    
    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2
    
    L_incoherent_const = np.zeros((9, 9), dtype=complex)
    for L, L_dag_L in [(L1, L1_dag_L1), (L2, L2_dag_L2)]:
        term1 = np.kron(L, L.conj())
        term2 = 0.5 * (np.kron(L_dag_L, np.eye(3)) + np.kron(np.eye(3), L_dag_L.T))
        L_incoherent_const += (term1 - term2)

    def build_liouvillian_fast(H):
        H_kron_I = np.kron(H, np.eye(3))
        I_kron_H_T = np.kron(np.eye(3), H.T)
        L_coherent = -1j * (H_kron_I - I_kron_H_T)
        return L_coherent + L_incoherent_const

    def build_hamiltonian_fast(I, Q, delta, scale):
        H_det = delta * n
        H_anh = 0.5 * alpha * n @ n_minus_1
        H_drive = 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
        return H_det + H_anh + H_drive
