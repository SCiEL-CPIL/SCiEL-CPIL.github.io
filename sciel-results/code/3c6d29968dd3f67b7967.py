            violated_k4s = get_violations(colours)
            while len(violated_k4s) > 0 and time.time() < deadline:
                # Pick a random violated K4
                k4_idx = violated_k4s[random.randint(0, len(violated_k4s)-1)]
                # Pick a random edge in this K4
                edge_indices_in_k4 = k4_edges_np[k4_idx]
                eidx = edge_indices_in_k4[random.randint(0, 5)]
                
                old_color = colours[eidx]
                new_color = random.randint(0, target_colors - 1)
                if new_color == old_color:
                    continue
                
                colours[eidx] = new_color
                new_violated_k4s = get_violations(colours)
                
                if len(new_violated_k4s) <= len(violated_k4s):
                    violated_k4s = new_violated_k4s
                else:
                    colours[eidx] = old_color
