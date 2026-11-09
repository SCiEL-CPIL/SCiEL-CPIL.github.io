    def simulate(pulse_I, pulse_Q, detuning, scale, init_state):
        rho0 = np.outer(init_state, init_state.conj())
        
        def ode_func(t_val, rho_flat):
            rho = rho_flat.reshape(3, 3)
            idx = int(t_val / dt)
            if idx >= N:
                idx = N - 1
            I_val = pulse_I[idx]
            Q_val = pulse_Q[idx]
            return liouvillian(rho, t_val, (detuning, scale, I_val, Q_val)).flatten()
        
        sol = solve_ivp(ode_func, [0, N*dt], rho0.flatten(), method='DOP853', t_eval=t, rtol=1e-9, atol=1e-12)
        if not sol.success:
            return 0.0
        rho_final = sol.y[:, -1].reshape(3, 3)
        
        # Ideal X_pi unitary in 3-level space
        U_ideal = np.zeros((3,3), dtype=complex)
        U_ideal[0,1] = -1j
        U_ideal[1,0] = -1j
        
        target_state = U_ideal @ init_state
        target_rho = np.outer(target_state, target_state.conj())
        fidelity = np.real(np.trace(rho_final @ target_rho))
        return fidelity
