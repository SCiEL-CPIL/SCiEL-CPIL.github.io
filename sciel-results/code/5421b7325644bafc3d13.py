import time

def design(problem):
    size = problem["pixels"]
    # Super-ellipse with optimized parameters
    # Previous attempt: n=4, a=0.27 (Score 0.01)
    # Seed: Circle r=0.2 (Score 1.0)
    # Hypothesis: The size 0.27 was too large. Let's try a smaller super-ellipse.
    # A super-ellipse with n=4 and a=0.21 is slightly "squarer" than a circle of r=0.21, 
    # but very close. It might capture a slightly larger effective area or shape benefit.
    
    n_exp = 4.0
    a = 0.21
    
    cell = [[0 for _ in range(size)] for _ in range(size)]
    
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            
            # Super-ellipse equation: |x/a|^n + |y/a|^n < 1
            # Using absolute values for symmetry
            val = (abs(x) / a) ** n_exp + (abs(y) / a) ** n_exp
            if val < 1.0:
                cell[i][j] = 1
            else:
                cell[i][j] = 0
                
    return cell
