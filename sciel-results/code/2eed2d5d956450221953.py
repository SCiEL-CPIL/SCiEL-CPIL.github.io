<<<<<<< SEARCH
import numpy as np
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 580.0  # Safety margin for 600s limit
    
    # Candidate side lengths for the 10 squares.
    # We try to maximize sum(s).
    # A common high-performing structure for 10 squares is a mix of sizes.
    # Let's generate a few candidate "size profiles" and layouts.
    
    # Profile 1: 4 large, 6 small (2x2 grid of large, surrounding small?)
    # Profile 2: 2 large, 8 small
    # Profile 3: 10 equal (baseline)
    # Profile 4: 3 large, 7 small
    
    # We will define specific layouts for these profiles and then refine.
    
    def is_valid(cx, cy, s, tol=1e-7):
=======
import numpy as np
import time
from scipy.optimize import linprog

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 580.0  # Safety margin for 600s limit
    
    # Candidate side lengths for the 10 squares.
    # We try to maximize sum(s).
    # A common high-performing structure for 10 squares is a mix of sizes.
    # Let's generate a few candidate "size profiles" and layouts.
    
    # Profile 1: 4 large, 6 small (2x2 grid of large, surrounding small?)
    # Profile 2: 2 large, 8 small
    # Profile 3: 10 equal (baseline)
    # Profile 4: 3 large, 7 small
    
    # We will define specific layouts for these profiles and then refine.
    
    def is_valid(cx, cy, s, tol=1e-7):
>>>>>>> REPLACE

