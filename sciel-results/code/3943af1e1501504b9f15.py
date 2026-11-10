                # Pick a random violation with lazy deletion
                k4_idx = -1
                k5_idx = -1
                violation_type = ''
                
                if violating_k4s:
                    # Try to find a valid K4
                    for _ in range(10):
                        k4_idx = random.choice(violating_k4s)
                        if red_k4_counts[k4_idx] == 4:
                            violation_type = 'K4'
                            break
                        else:
                            violating_k4s.remove(k4_idx) # Clean up
                
                if k4_idx == -1 and violating_k5s:
                    for _ in range(10):
                        k5_idx = random.choice(violating_k5s)
                        if blue_k5_counts[k5_idx] == 10:
                            violation_type = 'K5'
                            break
                        else:
                            violating_k5s.remove(k5_idx)
                            
                if violation_type == '':
                    # No valid violations found, but total_violations > 0?
                    # Re-scan?
                    # This should be rare.
                    break
