        best_energy = energy
        best_colors = colors[:]
        
        # Run SA for a fixed number of steps or time
        # We have ~100s left after cycle generation.
        # Let's run for 50s.
        sa_start = time.time()
        
        while time.time() < sa_start + 50.0:
             # ... SA steps ...
             if energy < best_energy:
                 best_energy = energy
                 best_colors = colors[:]
                 if energy == 0:
                     return best_colors
             
             # Reheat if stuck?
             # If no improvement in last 1000 steps, reheat
             
        return best_colors if best_energy == 0 else None
