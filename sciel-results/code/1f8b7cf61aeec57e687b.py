        # Flip
        idx = np.random.randint(n)
        c[idx] *= -1
        
        new_val = np.max(np.abs(np.fft.fft(c)))
        
        delta = new_val - current_val
        accept = False
        if delta < 0:
            accept = True
        else:
            if delta < 50 * T:
                if np.random.rand() < np.exp(-delta / T):
                    accept = True
            
        if accept:
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_c = c.copy()
                if best_val / np.sqrt(n) <= target_raw + 1e-6:
                    return best_c.tolist()
        else:
            # Revert
            c[idx] *= -1
