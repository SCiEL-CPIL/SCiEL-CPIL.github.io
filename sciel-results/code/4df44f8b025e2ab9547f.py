                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
                    if best_gain == 35: # Max possible for k=7, t=3
                        break
