                # 2-Swap (First Improvement)
                if not improved:
                    in_list = list(current_set)
                    out_list = [i for i in range(n) if i not in current_set]
                    # Sort out_list by value descending to find improvements faster
                    out_list.sort(key=lambda x: reciprocals[x], reverse=True)
                    
                    found = False
                    for a_idx in range(len(in_list)):
                        i = in_list[a_idx]
                        for b_idx in range(a_idx + 1, len(in_list)):
                            k = in_list[b_idx]
                            sum_rem = reciprocals[i] + reciprocals[k]
                            base = current_sum - sum_rem
                            
                            for c_idx in range(len(out_list)):
                                j = out_list[c_idx]
                                for d_idx in range(c_idx + 1, len(out_list)):
                                    l = out_list[d_idx]
                                    new_sum = base + reciprocals[j] + reciprocals[l]
                                    if new_sum < 1 and new_sum > current_sum:
                                        current_set.remove(i)
                                        current_set.remove(k)
                                        current_set.add(j)
                                        current_set.add(l)
                                        current_sum = new_sum
                                        improved = True
                                        found = True
                                        break
                                if found: break
                            if found: break
                        if found: break
