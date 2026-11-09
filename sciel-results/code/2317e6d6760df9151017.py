                # Update neighbors
                old_v = 1 - labels[v]
                u_indices = neighbors[v]
                if len(u_indices) > 0:
                    was_same = (labels[u_indices] == old_v)
                    # If same, gain decreases by 2. If opposite, gain increases by 2.
                    delta_gains = np.where(was_same, -2, 2)
                    gains[u_indices] += delta_gains
