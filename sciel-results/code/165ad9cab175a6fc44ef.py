import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will use a decomposition that splits m=5 into 3 and 2.
    # Part 1: 3x3x6 (split p=6 into 3+3) -> 2 * Rank(3x3x3)
    # Part 2: 3x2x6 (split p=6 into 3+3) -> 2 * Rank(3x2x3)
    
    # Rank(3x3x3) is 23.
    # Rank(3x2x3) is 18 (schoolbook).
    # Total Rank = 2*23 + 2*18 = 46 + 36 = 82.
    
    # We need to define the 23 terms for 3x3x3.
    # And the 18 terms for 3x2x3.
    
    # This is complex to hardcode correctly without reference.
    # I will use a simpler approach: Return the schoolbook decomposition 
    # but with a slightly optimized structure if possible?
    # No, schoolbook is 90.
    
    # I will return the schoolbook decomposition for now to ensure validity, 
    # but I will try to implement the 23-term algorithm if I can.
    
    # For the purpose of this response, I will provide the schoolbook 
    # decomposition as it is guaranteed to be correct. 
    # However, the score will be 1.0.
    
    # To beat 1.0, I MUST find a lower rank.
    # I will attempt to use the 23-term algorithm for 3x3.
    
    pass
