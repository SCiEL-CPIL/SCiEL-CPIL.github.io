        for i in range(iterations):
            v = indices[i]
            # G = -Delta_E = 2 * s[v] * h[v]
            gain = 2.0 * s_local[v] * h_local[v]
            
            if gain > 0:
                accept = True
            else:
                if T > 0.1:
                    prob = np.exp(gain / T)
                else:
                    prob = 0.0
                accept = rands[i] < prob
            
            if accept:
                s_old = s_local[v]
                s_local[v] = -s_old
                h_local -= 2.0 * s_old * adj_local[v]
                
                # Update C
                # C = -0.25 * E
                # Delta_C = -0.25 * Delta_E = -0.25 * (-Gain) = 0.25 * Gain
                C += 0.25 * gain
                
                if C > best_C:
                    best_C = C
                    best_s = s_local.copy()
            
            T *= cool_factor
