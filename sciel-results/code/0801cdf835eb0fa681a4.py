    def get_violators(cand, current_arr):
        # current_arr is numpy array of int64
        # Convert to list of ints for faster bit_count? 
        # Or iterate over numpy array
        violators = []
        for i, c in enumerate(current_arr):
            if (cand ^ c).bit_count() < _MIN_D:
                violators.append(i)
        return violators
