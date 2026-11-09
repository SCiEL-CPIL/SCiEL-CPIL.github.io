    # ... inside loop ...
            i = np.random.randint(N)
            old_pt = points[i].copy()
            points[i] += delta
            
            new_energy = energy(points)
            
            if new_energy < current_energy:
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_points = points.copy()
            else:
                delta_e = new_energy - current_energy
                accept = False
                if delta_e <= 0:
                    accept = True
                else:
                    if delta_e / T < 500:
                        accept = np.random.random() < np.exp(-delta_e / T)
                
                if accept:
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_points = points.copy()
                else:
                    points[i] = old_pt
