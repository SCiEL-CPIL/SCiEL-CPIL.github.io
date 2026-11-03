            # ...
            redundant_indices = []
            for i, block in enumerate(blocks):
                # ...
                if not essential:
                    redundant_indices.append(i)
            
            if redundant_indices:
                changed = True
                # Remove redundant blocks
                blocks = [b for i, b in enumerate(blocks) if i not in set(redundant_indices)]
