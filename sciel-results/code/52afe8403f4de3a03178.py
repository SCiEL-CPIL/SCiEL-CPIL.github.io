  for t in range(steps - 1, -1, -1):
      for qi in range(2 * Q + 1):
          q = qi - Q
          
          # Get V_next for neighbors
          # q-1 index: qi-1. If qi==0, q=-Q, no bid fill.
          # q+1 index: qi+1. If qi==2Q, q=Q, no ask fill.
          
          V_q = V_next[qi]
          
          if qi == 0: # q = -Q
              # No ask fill
              P_a_eff = np.zeros(num_depths)
              P_b_eff = P_b
          elif qi == 2 * Q: # q = Q
              # No bid fill
              P_b_eff = np.zeros(num_depths)
              P_a_eff = P_a
          else:
              P_b_eff = P_b
              P_a_eff = P_a
          
          V_m = V_next[qi - 1] if qi > 0 else 0.0
          V_p = V_next[qi + 1] if qi < 2 * Q else 0.0
          
          # Compute values for all pairs (i, j)
          # i is bid depth index, j is ask depth index
          
          # P_b[i] * d_b[i] + P_a[j] * d_a[j] + ...
          # We can do this with outer products or meshgrid.
          
          Pb = P_b_eff[:, np.newaxis] # K, 1
          Pa = P_a_eff[np.newaxis, :] # 1, K
          db = depths[:, np.newaxis]
          da = depths[np.newaxis, :]
          
          # Term 1: P_b * d_b
          t1 = Pb * db # K, 1
          # Term 2: P_a * d_a
          t2 = Pa * da # 1, K
          
          # Term 3: P_b * (1 - P_a) * V_m
          t3 = Pb * (1 - Pa) * V_m # K, K
          
          # Term 4: P_a * (1 - P_b) * V_p
          t4 = Pa * (1 - Pb) * V_p # K, K
          
          # Term 5: (1 - P_b) * (1 - P_a) * V_q
          t5 = (1 - Pb) * (1 - Pa) * V_q # K, K
          
          # Total expected value from next step
          V_next_expected = t1 + t2 + t3 + t4 + t5
          
          # Add current penalty
          # Penalty is -0.5 * gamma * sigma^2 * q^2 * dt
          # This is constant for this q, so it doesn't affect argmax.
          # But it affects V_t(q) stored.
          
          max_val = np.max(V_next_expected)
          flat_idx = np.argmax(V_next_expected)
          i_opt, j_opt = np.unravel_index(flat_idx, (num_depths, num_depths))
          
          V_curr[qi] = -0.5 * gamma * sigma**2 * q**2 * dt + max_val
          
          bid_opt[t, qi] = i_opt
          ask_opt[t, qi] = j_opt
          
      V_next = V_curr
  