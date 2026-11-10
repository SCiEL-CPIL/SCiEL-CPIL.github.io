<<<<<<< SEARCH
            # Add more 2-norm vectors
            for v in candidates_2:
                if time.time() > deadline - 30:
                    break
                if v in sel2:
                    continue
=======
            # Add more 2-norm vectors
            for v in candidates_2:
                if time.time() > deadline - 30:
                    break
                if vec_in_list(v, sel2):
                    continue
>>>>>>> REPLACE
