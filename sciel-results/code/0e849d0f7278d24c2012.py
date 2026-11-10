<<<<<<< SEARCH
    # Final high-precision polish pass
    # Use more sampling points and a tighter local search
    S_polish = 10000
    angles_polish = 2 * np.pi * np.arange(S_polish) / S_polish
    z_powers_polish = np.zeros((n, S_polish), dtype=complex)
    for k in range(n):
        z_powers_polish[k] = np.exp(1j * k * angles_polish)
    
    seq = list(best_seq_copy)
    P_vals_polish = np.sum(np.array(seq)[:, None] * z_powers_polish, axis=0)
    current_peak_polish = get_peak(P_vals_polish)
    best_peak_polish = current_peak_polish
    best_seq_polish = list(seq)
    
    # Tight local search with smaller temperature
    T = 0.05
    T_min_polish = 1e-7
    alpha_polish = 0.9985
        
    # Track consecutive non-improvements for adaptive step
    no_improve_count = 0
        
    while time.time() < deadline - 2 and T > T_min_polish:
        idx = random.randint(0, n - 1)
        c = seq[idx]
        delta_val = -2 * c * z_powers_polish[idx]
            
        new_P_vals = P_vals_polish + delta_val
        new_peak = get_peak(new_P_vals)
            
        if new_peak < current_peak_polish:
            P_vals_polish = new_P_vals
            current_peak_polish = new_peak
            seq[idx] = -seq[idx]
            no_improve_count = 0
                
            if current_peak_polish < best_peak_polish:
                best_peak_polish = current_peak_polish
                best_seq_polish = list(seq)
        else:
            delta_peak = new_peak - current_peak_polish
            if random.random() < exp(-delta_peak / T):
                P_vals_polish = new_P_vals
                current_peak_polish = new_peak
                seq[idx] = -seq[idx]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
        T *= alpha_polish
            
        # If stuck, do a small random kick
        if no_improve_count > 200:
            for _ in range(random.randint(1, 3)):
                kick_idx = random.randint(0, n - 1)
                seq[kick_idx] = -seq[kick_idx]
                P_vals_polish = np.sum(np.array(seq)[:, None] * z_powers_polish, axis=0)
                current_peak_polish = get_peak(P_vals_polish)
            no_improve_count = 0
    
    # Return the best sequence found
    return best_seq_polish
=======
    # Final high-precision polish pass
    # Use more sampling points and a tighter local search
    S_polish = 12000
    angles_polish = 2 * np.pi * np.arange(S_polish) / S_polish
    z_powers_polish = np.zeros((n, S_polish), dtype=complex)
    for k in range(n):
        z_powers_polish[k] = np.exp(1j * k * angles_polish)
    
    seq = list(best_seq_copy)
    P_vals_polish = np.sum(np.array(seq)[:, None] * z_powers_polish, axis=0)
    current_peak_polish = get_peak(P_vals_polish)
    best_peak_polish = current_peak_polish
    best_seq_polish = list(seq)
    
    # Deterministic flip-all hill climbing + stochastic SA polish
    # Alternate between: (1) try flipping each coefficient, accept if improves
    #                    (2) stochastic SA step for exploration
    max_no_improve = 0
    no_improve_streak = 0
    
    while time.time() < deadline - 3:
        # Phase 1: Deterministic single-flip sweep
        improved = False
        for idx in range(n):
            if time.time() >= deadline - 3:
                break
            c = seq[idx]
            delta_val = -2 * c * z_powers_polish[idx]
            new_P_vals = P_vals_polish + delta_val
            new_peak = get_peak(new_P_vals)
            if new_peak < current_peak_polish - 1e-12:
                P_vals_polish = new_P_vals
                current_peak_polish = new_peak
                seq[idx] = -seq[idx]
                improved = True
                if current_peak_polish < best_peak_polish:
                    best_peak_polish = current_peak_polish
                    best_seq_polish = list(seq)
        
        if not improved:
            no_improve_streak += 1
            max_no_improve = max(max_no_improve, no_improve_streak)
        else:
            no_improve_streak = 0
        
        # Phase 2: Stochastic SA steps for escape
        if no_improve_streak >= 1:
            T = 0.03 / (1 + no_improve_streak * 0.5)
            n_sa_steps = min(50, 20 + no_improve_streak * 10)
            for _ in range(n_sa_steps):
                if time.time() >= deadline - 3:
                    break
                idx = random.randint(0, n - 1)
                c = seq[idx]
                delta_val = -2 * c * z_powers_polish[idx]
                new_P_vals = P_vals_polish + delta_val
                new_peak = get_peak(new_P_vals)
                if new_peak < current_peak_polish:
                    P_vals_polish = new_P_vals
                    current_peak_polish = new_peak
                    seq[idx] = -seq[idx]
                    no_improve_streak = 0
                    if current_peak_polish < best_peak_polish:
                        best_peak_polish = current_peak_polish
                        best_seq_polish = list(seq)
                else:
                    delta_peak = new_peak - current_peak_polish
                    if random.random() < exp(-delta_peak / T):
                        P_vals_polish = new_P_vals
                        current_peak_polish = new_peak
                        seq[idx] = -seq[idx]
                    else:
                        no_improve_streak += 1
        
        # If stuck for a long time, do a multi-flip kick
        if no_improve_streak > 30:
            n_kicks = random.randint(2, 5)
            for _ in range(n_kicks):
                kick_idx = random.randint(0, n - 1)
                seq[kick_idx] = -seq[kick_idx]
            P_vals_polish = np.sum(np.array(seq)[:, None] * z_powers_polish, axis=0)
            current_peak_polish = get_peak(P_vals_polish)
            no_improve_streak = 0
    
    # Final deterministic sweep on best found
    seq = list(best_seq_polish)
    P_vals_final = np.sum(np.array(seq)[:, None] * z_powers_polish, axis=0)
    peak_final = get_peak(P_vals_final)
    for _ in range(5):
        for idx in range(n):
            c = seq[idx]
            delta_val = -2 * c * z_powers_polish[idx]
            new_P_vals = P_vals_final + delta_val
            new_peak = get_peak(new_P_vals)
            if new_peak < peak_final - 1e-12:
                P_vals_final = new_P_vals
                peak_final = new_peak
                seq[idx] = -seq[idx]
    
    if peak_final < best_peak_polish:
        best_seq_polish = list(seq)
    
    # Return the best sequence found
    return best_seq_polish
>>>>>>> REPLACE

