     if old_color == 0:
         k4_red_counts[k4_i] -= 1
         if k4_red_counts[k4_i] == 3: # Was 4
             active_k4.discard(k4_i)
     else:
         k4_red_counts[k4_i] += 1
         if k4_red_counts[k4_i] == 4: # Was 3
             active_k4.add(k4_i)
     