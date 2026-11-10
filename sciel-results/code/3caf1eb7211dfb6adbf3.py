<<<<<<< SEARCH
def _local_search(points, max_iters=5000):
    """
    Local search to maximize the number of triples.
    """
    current_points = [list(p) for p in points]
    current_triples = _count_triples(current_points)
    best_points = [list(p) for p in current_points]
    best_triples = current_triples
    
    for i in range(max_iters):
        if _elapsed() > _DEADLINE - 10:
            break
        
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_x, old_y = current_points[idx]
        
        # Try a few random new positions
        found_improvement = False
        for _ in range(10):
            # Move in a small neighborhood
            new_x = random.randint(old_x - 5, old_x + 5)
            new_y = random.randint(old_y - 5, old_y + 5)
            
            # Check bounds
            if not (-100 <= new_x <= 100 and -100 <= new_y <= 100):
                continue
                
            # Check distinctness
            if (new_x, new_y) == (old_x, old_y):
                continue
            is_duplicate = False
            for j in range(20):
                if j != idx and current_points[j] == [new_x, new_y]:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
                
            # Try the move
            new_points = [list(p) for p in current_points]
            new_points[idx] = [new_x, new_y]
            
            # Check validity (no 4 collinear)
            if not _check_no_four_collinear(new_points):
                continue
                
            # Count triples
            new_triples = _count_triples(new_points)
            
            if new_triples > current_triples:
                current_points = new_points
                current_triples = new_triples
                if current_triples > best_triples:
                    best_triples = current_triples
                    best_points = [list(p) for p in current_points]
                found_improvement = True
                break
            elif new_triples == current_triples and random.random() < 0.1:
                # Simulated annealing-like move to escape local optima
                # Accept equal or slightly worse? No, we want to maximize triples.
                # Let's just stick to strict improvement or random walk
                current_points = new_points
                current_triples = new_triples
                found_improvement = True
                break
                
        if not found_improvement and i % 100 == 0:
            # Random restart of one point to a random location to shake up
            idx = random.randint(0, 19)
            new_x = random.randint(-20, 20)
            new_y = random.randint(-20, 20)
            if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                is_dup = False
                for j in range(20):
                    if j != idx and current_points[j] == [new_x, new_y]:
                        is_dup = True
                        break
                if not is_dup:
                    new_points = [list(p) for p in current_points]
                    new_points[idx] = [new_x, new_y]
                    if _check_no_four_collinear(new_points):
                        current_points = new_points
                        current_triples = _count_triples(current_points)
                        
    return best_points
=======
def _local_search(points, max_iters=8000):
    """
    Local search to maximize the number of triples.
    """
    current_points = [list(p) for p in points]
    current_triples = _count_triples(current_points)
    best_points = [list(p) for p in current_points]
    best_triples = current_triples
    
    for i in range(max_iters):
        if _elapsed() > _DEADLINE - 15:
            break
        
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_x, old_y = current_points[idx]
        
        # Try multiple random new positions
        found_improvement = False
        num_tries = 15 if i < max_iters // 2 else 8
        for _ in range(num_tries):
            # Move in a neighborhood, larger in early iterations
            radius = 6 if i < max_iters // 3 else 4
            new_x = random.randint(old_x - radius, old_x + radius)
            new_y = random.randint(old_y - radius, old_y + radius)
            
            # Check bounds
            if not (-100 <= new_x <= 100 and -100 <= new_y <= 100):
                continue
                
            # Check distinctness
            if (new_x, new_y) == (old_x, old_y):
                continue
            is_duplicate = False
            for j in range(20):
                if j != idx and current_points[j][0] == new_x and current_points[j][1] == new_y:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
                
            # Try the move
            new_points = [list(p) for p in current_points]
            new_points[idx] = [new_x, new_y]
            
            # Check validity (no 4 collinear)
            if not _check_no_four_collinear(new_points):
                continue
                
            # Count triples
            new_triples = _count_triples(new_points)
            
            if new_triples > current_triples:
                current_points = new_points
                current_triples = new_triples
                if current_triples > best_triples:
                    best_triples = current_triples
                    best_points = [list(p) for p in current_points]
                found_improvement = True
                break
            elif new_triples == current_triples and random.random() < 0.05:
                current_points = new_points
                current_triples = new_triples
                found_improvement = True
                break
                
        if not found_improvement and i % 50 == 0:
            # Random restart of one point to a random location to shake up
            idx = random.randint(0, 19)
            new_x = random.randint(-15, 15)
            new_y = random.randint(-15, 15)
            is_dup = False
            for j in range(20):
                if j != idx and current_points[j][0] == new_x and current_points[j][1] == new_y:
                    is_dup = True
                    break
            if not is_dup:
                new_points = [list(p) for p in current_points]
                new_points[idx] = [new_x, new_y]
                if _check_no_four_collinear(new_points):
                    current_points = new_points
                    current_triples = _count_triples(current_points)
                        
    return best_points
>>>>>>> REPLACE

