        def get_strassen_terms(n):
            # Returns list of (u, v, w) for n x n x n Strassen
            # u, v, w are lists of length n*n
            if n == 1:
                return [([1.0], [1.0], [1.0])]
            
            n_half = n // 2
            terms = []
            
            # Define 7 products
            # Each product is a pair of (u_linear, v_linear) for the sub-blocks
            # And we need to combine them with the C reconstruction
            
            # This is hard to do with just vectors.
            # We need to know how the sub-blocks map to the global block.
            
            # Let's use the linear form approach.
            # Return list of (u_form, v_form, w_form)
            # where u_form is a dict {(i,k): coeff}
            
            pass
        