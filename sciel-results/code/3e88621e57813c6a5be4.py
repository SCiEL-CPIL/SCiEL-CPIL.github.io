"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 12
    selected = []
    current_sums = {0}
    
    for _ in range(n):
        # Try candidates starting from 1
        candidate = 1
        while True:
            # Check if candidate is distinct from already selected
            if candidate in selected:
                candidate += 1
                continue
            
            # Calculate new sums
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + candidate)
            
            # Check if all sums are distinct
            if len(new_sums) == len(current_sums) * 2:
                # Valid
                selected.append(candidate)
                current_sums = new_sums
                break
            else:
                candidate += 1
                
    return selected
# EVOLVE-BLOCK-END
