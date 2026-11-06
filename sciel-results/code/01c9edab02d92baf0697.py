    def construct():
        import itertools
        
        points = list(range(12))
        all_blocks = list(itertools.combinations(points, 6))
        block_subs = {b: list(itertools.combinations(b, 5)) for b in all_blocks}
        
        # Heuristic: Order blocks by some criterion?
        # Lexicographic is fine.
        
        first_block = (0, 1, 2, 3, 4, 5)
        covered = set(block_subs[first_block])
        blocks = [first_block]
        
        # Candidates: blocks that don't conflict with first_block
        candidates = [b for b in all_blocks if b != first_block and not any(s in covered for s in block_subs[b])]
        
        # To make backtracking faster, we can sort candidates.
        # A good heuristic: Blocks that cover more "rare" subsets?
        # For now, lexicographic.
        
        def backtrack(idx, covered, blocks):
            if len(blocks) == 132:
                return True
            
            for i in range(idx, len(candidates)):
                b = candidates[i]
                # Check conflict
                conflict = False
                for s in block_subs[b]:
                    if s in covered:
                        conflict = True
                        break
                if conflict:
                    continue
                
                new_covered = covered.union(block_subs[b])
                new_blocks = blocks + [b]
                
                if backtrack(i + 1, new_covered, new_blocks):
                    return True
            
            return False
        
        if backtrack(0, covered, blocks):
            return [list(b) for b in blocks]
        else:
            # Fallback
            covered_g, blocks_g = set(), []
            for b in all_blocks:
                subs = block_subs[b]
                if not any(s in covered_g for s in subs):
                    blocks_g.append(list(b))
                    covered_g.update(subs)
            return blocks_g
    