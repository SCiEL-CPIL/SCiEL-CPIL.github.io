import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 250 # 50s margin for 300s limit? No, 330s limit. Let's use 300s to be safe.

    def __call__(self, func):
        # ... implementation ...
        return best_x, best_f
