<<<<<<< SEARCH
                if verify_and_update(allocations, payments):
                    revenue = np.dot(probs, payments)
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_allocations = np.copy(allocations)
                        best_payments = np.copy(payments)

        except Exception:
            pass

    if time_check():
        return {'allocations': best_allocations, 'payments': best_payments}

    # --- Approach 3: Direct threshold/reserve price search ---
=======
                if verify_and_update(allocations, payments):
                    revenue = float(np.dot(probs, payments))
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_allocations = np.copy(allocations)
                        best_payments = np.copy(payments)

        except Exception:
            pass

    if time_check():
        return {'allocations': best_allocations, 'payments': best_payments}

    # --- Approach 3: Direct threshold/reserve price search ---
>>>>>>> REPLACE

