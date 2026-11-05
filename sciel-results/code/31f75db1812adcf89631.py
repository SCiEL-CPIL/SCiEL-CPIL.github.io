import time

def construct():
    deadline = time.time() + 85
    
    def get_new_sums(values, x):
        """Get new consecutive sums when appending x to values."""
        total = x
        new_sums = [x]
        for a in reversed(values):
            total += a
            new_sums.append(total)
        return new_sums
    
    def is_valid(values, sums, x):
        """Check if x can be appended to values without sum collision."""
        new_sums = get_new_sums(values, x)
        for s in new_sums:
            if s in sums:
                return False
        return True
    
    # Beam search
    beam_width = 50
    candidates = []  # list of (values, sums_set)
    
    # Start with empty
    candidates.append(([], set()))
    
    best = []
    
    while candidates and time.time() < deadline:
        next_candidates = []
        for values, sums in candidates:
            if len(values) > len(best):
                best = list(values)
            
            # Try to extend
            start = values[-1] + 1 if values else 1
            found = False
            for x in range(start, 257):
                if time.time() > deadline:
                    break
                if is_valid(values, sums, x):
                    new_sums = get_new_sums(values, x)
                    new_sums_set = set(new_sums)
                    all_sums = sums | new_sums_set
                    next_candidates.append((values + [x], all_sums))
                    found = True
                    # Don't break - try all valid x to build a diverse beam
            if not found:
                # Dead end, don't extend
                pass
        
        if not next_candidates:
            break
        
        # Keep only the best beam_width candidates
        # Sort by length (longer is better), then by last element (smaller last element is better for future extension)
        next_candidates.sort(key=lambda c: (-len(c[0]), c[0][-1] if c[0] else 0))
        candidates = next_candidates[:beam_width]
    
    return best
