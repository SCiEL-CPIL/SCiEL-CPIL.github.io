            # 3. Try Remove 1, Add 1 (Swap)
            # Only if no Add and no Remove-1-Add-2 was found in this iteration?
            # Actually, if we found a Remove-1-Add-2, we already set improved=True and will break.
            # So if we reach here, improved is still False.
            
            # Try swaps
            vertices_in_sel = [v for v in range(n) if (sel_mask >> v) & 1]
            for u in vertices_in_sel:
                old_sel_mask = sel_mask
                old_sel_pairs_mask = sel_pairs_mask
                
                sel_mask &= ~(1 << u)
                sel_pairs_mask &= ~pairs_involving_mask[u]
                
                for v in range(n):
                    if not ((sel_mask >> v) & 1):
                        if is_valid_addition(v, sel_pairs_mask):
                            # Valid swap
                            sel_mask = old_sel_mask | (1 << v)
                            sel_pairs_mask = old_sel_pairs_mask | pairs_involving_mask[v]
                            improved = True
                            break
                if improved:
                    break
                # Restore if no swap found for this u
                if not improved:
                    sel_mask = old_sel_mask
                    sel_pairs_mask = old_sel_pairs_mask
