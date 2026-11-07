        def sbx_crossover(x1, x2):
            """Simulated Binary Crossover."""
            if np.random.random() > p_c:
                return x1.copy(), x2.copy()
            
            x1_new = x1.copy()
            x2_new = x2.copy()
            
            for i in range(dim):
                if np.random.random() >= 0.5:
                    continue
                
                x1_i = x1[i]
                x2_i = x2[i]
                
                if abs(x1_i - x2_i) < 1e-14:
                    continue
                
                # Calculate beta
                if x1_i > x2_i:
                    beta = 1.0 + 2.0 * (x1_i - lo[i]) / (x1_i - x2_i)
                else:
                    beta = 1.0 + 2.0 * (hi[i] - x2_i) / (x2_i - x1_i)
                
                # Calculate beta_q
                if np.random.random() < 0.5:
                    beta_q = (0.5 * (1 + (1 + (eta_c + 1) * (2 * np.random.random() - 1)) * (beta - 1))) ** (1.0 / (eta_c + 1))
                else:
                    beta_q = (0.5 * (1 - (1 - (eta_c + 1) * (2 * np.random.random() - 1)) * (beta - 1))) ** (1.0 / (eta_c + 1))
                
                # Calculate children
                x1_new[i] = 0.5 * ((1 + beta_q) * x1_i + (1 - beta_q) * x2_i)
                x2_new[i] = 0.5 * ((1 - beta_q) * x1_i + (1 + beta_q) * x2_i)
                
                # Clip
                x1_new[i] = np.clip(x1_new[i], lo[i], hi[i])
                x2_new[i] = np.clip(x2_new[i], lo[i], hi[i])
                
            return x1_new, x2_new
