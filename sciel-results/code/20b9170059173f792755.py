    def local_search(blocks, seed):
        random.seed(seed)
        current_blocks = list(blocks)
        current_set = set(current_blocks)
        current_mask = 0
        for idx in current_blocks:
            current_mask |= block_masks[idx]
        current_defect = (full_mask ^ current_mask).bit_count()
        current_score = current_defect * 10000 + len(current_blocks)
        
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() < deadline:
            if no_improve_count > max_no_improve:
                break
            
            move_type = random.choice(['del', 'add', 'swap'])
            
            if move_type == 'del' and len(current_blocks) > 1:
                i = random.randint(0, len(current_blocks)-1)
                b_remove = current_blocks[i]
                
                new_mask = 0
                for j, idx in enumerate(current_blocks):
                    if j != i:
                        new_mask |= block_masks[idx]
                
                new_defect = (full_mask ^ new_mask).bit_count()
                new_score = new_defect * 10000 + (len(current_blocks)-1)
                
                if new_score < current_score:
                    current_blocks.pop(i)
                    current_set.remove(b_remove)
                    current_mask = new_mask
                    current_defect = new_defect
                    current_score = new_score
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
            elif move_type == 'add':
                while True:
                    b_idx = random.randint(0, len(all_blocks)-1)
                    if b_idx not in current_set:
                        break
                new_mask = current_mask | block_masks[b_idx]
                new_defect = (full_mask ^ new_mask).bit_count()
                new_score = new_defect * 10000 + (len(current_blocks)+1)
                
                if new_score < current_score:
                    current_blocks.append(b_idx)
                    current_set.add(b_idx)
                    current_mask = new_mask
                    current_defect = new_defect
                    current_score = new_score
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            elif move_type == 'swap' and len(current_blocks) > 0:
                i = random.randint(0, len(current_blocks)-1)
                b_old = current_blocks[i]
                
                while True:
                    b_idx = random.randint(0, len(all_blocks)-1)
                    if b_idx not in current_set:
                        break
                
                new_mask = 0
                for j, idx in enumerate(current_blocks):
                    if j != i:
                        new_mask |= block_masks[idx]
                new_mask |= block_masks[b_idx]
                
                new_defect = (full_mask ^ new_mask).bit_count()
                new_score = new_defect * 10000 + len(current_blocks)
                
                if new_score < current_score:
                    current_blocks[i] = b_idx
                    current_set.remove(b_old)
                    current_set.add(b_idx)
                    current_mask = new_mask
                    current_defect = new_defect
                    current_score = new_score
                    no_improve_count = 0
                else:
                    no_improve_count += 1

        return current_blocks
